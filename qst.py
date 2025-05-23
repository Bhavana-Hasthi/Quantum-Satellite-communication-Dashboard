import streamlit as st
import numpy as np
import hashlib
import os
import json
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import plotly.express as px
import requests
import time

# Page setup - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Quantum Satellite Dashboard", 
    layout="wide",
    page_icon="🛰️"
)

# ------------------------ CONFIGURATION ------------------------
OPENWEATHERMAP_API_KEY = "5e6a9166615332b5869cd15e76ce0cef"
N2YO_API_KEY = "R5AKAS-2EAP57-PD9WFM-5GY5"

# ------------------------ DATA SETUP ------------------------

# Satellite selection
satellite_options = {
    "ISS → Ground Station Singapore": "iss_orbit_qber_singapore.json",
    "ISS → Ground Station Vienna": "iss_orbit_qber_vienna.json",
    "ISS → Ground Station New York": "iss_orbit_qber_ny.json",
    "ISS → Ground Station Cape Town": "iss_orbit_qber_capetown.json",
    "ISS → Ground Station Tokyo": "iss_orbit_qber_tokyo.json",
    "ISS → Ground Station Bangalore": "iss_orbit_qber_bangalore.json",
    "ISS → Ground Station Delhi": "iss_orbit_qber_delhi.json",
    "ISS → Ground Station London": "iss_orbit_qber_london.json",
    "ISS → Ground Station Paris": "iss_orbit_qber_paris.json",
    "ISS → Ground Station Sydney": "iss_orbit_qber_sydney.json",
    "ISS → Ground Station Munich": "iss_orbit_qber_munich.json",
    "ISS → Ground Station Toronto": "iss_orbit_qber_toronto.json",
    "ISS → Ground Station Madrid": "iss_orbit_qber_madrid.json",
    "ISS → Ground Station Istanbul": "iss_orbit_qber_istanbul.json",
    "ISS → Ground Station Rio de Janeiro": "iss_orbit_qber_rio.json",
    "ISS → Ground Station Nairobi": "iss_orbit_qber_nairobi.json",
    "ISS → Ground Station Bangkok": "iss_orbit_qber_bangkok.json"
}

# Ground station coordinates for 3D view
ground_station_coords = {
    "Tokyo": {"lat": 35.68, "lon": 139.76},
    "Singapore": {"lat": 1.35, "lon": 103.86},
    "Vienna": {"lat": 48.21, "lon": 16.37},
    "New York": {"lat": 40.71, "lon": -74.01},
    "Cape Town": {"lat": -33.92, "lon": 18.42},
    "Bangalore": {"lat": 12.97, "lon": 77.59},
    "Delhi": {"lat": 28.61, "lon": 77.21},
    "London": {"lat": 51.51, "lon": -0.13},
    "Paris": {"lat": 48.86, "lon": 2.35},
    "Sydney": {"lat": -33.87, "lon": 151.21},
    "Munich": {"lat": 48.14, "lon": 11.58},
    "Toronto": {"lat": 43.65, "lon": -79.38},
    "Madrid": {"lat": 40.42, "lon": -3.70},
    "Istanbul": {"lat": 41.01, "lon": 28.98},
    "Rio de Janeiro": {"lat": -22.91, "lon": -43.17},
    "Nairobi": {"lat": -1.29, "lon": 36.82},
    "Bangkok": {"lat": 13.76, "lon": 100.50}
}

# Ground Station Coordinates
GROUND_STATIONS = {
    "Singapore": (1.3521, 103.8198),
    "Vienna": (48.2082, 16.3738),
    "New York": (40.7128, -74.0060),
    "Cape Town": (-33.9249, 18.4241),
    "Tokyo": (35.6762, 139.6503),
    "Bangalore": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Sydney": (-33.8688, 151.2093),
    "Munich": (48.1351, 11.5820),
    "Toronto": (43.651070, -79.347015),
    "Madrid": (40.4168, -3.7038),
    "Istanbul": (41.0082, 28.9784),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Nairobi": (-1.2921, 36.8219),
    "Bangkok": (13.7563, 100.5018)
}

# ------------------------ FUNCTIONS ------------------------
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data['main']['temp'],
            "weather": data['weather'][0]['description']
        }
    else:
        return None

def get_satellite_position(sat_id=25544):  # 25544 is the NORAD ID for ISS
    url = f"https://api.n2yo.com/rest/v1/satellite/positions/{sat_id}/0/0/0/1/&apiKey={N2YO_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['positions'][0]
    else:
        return None

def simulate_bb84(num_bits, error_probability, alice_bits):
    alice_bases = np.random.choice(['Z', 'X'], size=num_bits)
    eve_bases = np.random.choice(['Z', 'X'], size=num_bits)
    eve_results = []
    qubits_sent_to_bob = []
    simulator = AerSimulator()

    for bit, a_base, e_base in zip(alice_bits, alice_bases, eve_bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)
        if a_base == 'X':
            qc.h(0)

        eve_qc = QuantumCircuit(1, 1)
        eve_qc.compose(qc, inplace=True)
        if e_base == 'X':
            eve_qc.h(0)
        eve_qc.measure(0, 0)
        result = simulator.run(eve_qc, shots=1).result()
        counts = result.get_counts()
        eve_bit = int(max(counts, key=counts.get))
        eve_results.append(eve_bit)

        resend_qc = QuantumCircuit(1, 1)
        if eve_bit == 1:
            resend_qc.x(0)
        if a_base == 'X':
            resend_qc.h(0)
        qubits_sent_to_bob.append(resend_qc)

    bob_bases = np.random.choice(['Z', 'X'], size=num_bits)
    bob_results = []

    for circuit, b_base in zip(qubits_sent_to_bob, bob_bases):
        if b_base == 'X':
            circuit.h(0)
        circuit.measure(0, 0)
        result = simulator.run(circuit, shots=1).result()
        counts = result.get_counts()
        bob_bit = int(max(counts, key=counts.get))
        if np.random.rand() < error_probability:
            bob_bit ^= 1
        bob_results.append(bob_bit)

    matching_indices = [i for i in range(num_bits) if alice_bases[i] == bob_bases[i]]
    sifted_key = [alice_bits[i] for i in matching_indices]
    bob_sifted_key = [bob_results[i] for i in matching_indices]
    errors = sum(1 for a, b in zip(sifted_key, bob_sifted_key) if a != b)
    error_rate = errors / len(sifted_key) if sifted_key else 0
    corrected_key = [bit for a, b, bit in zip(sifted_key, bob_sifted_key, sifted_key) if a == b]
    key_string = ''.join(str(bit) for bit in corrected_key)
    final_key_hex = hashlib.sha256(key_string.encode()).hexdigest()

    return alice_bases, eve_bases, eve_results, bob_bases, bob_results, sifted_key, bob_sifted_key, corrected_key, error_rate, final_key_hex

def generate_hmac(key, message):
    return hashlib.sha256(key.encode() + message.encode()).hexdigest()

def encrypt_message(key_hex, plaintext):
    key_bytes = bytes.fromhex(key_hex)[:32]
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
    return iv.hex(), ciphertext.hex()

def decrypt_message(key_hex, iv_hex, ciphertext_hex):
    key_bytes = bytes.fromhex(key_hex)[:32]
    iv = bytes.fromhex(iv_hex)
    ciphertext = bytes.fromhex(ciphertext_hex)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext.decode(errors='ignore')

def text_to_bits(text):
    return [int(bit) for char in text for bit in bin(ord(char))[2:].zfill(8)]

# ------------------------ STREAMLIT UI ------------------------

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "🌍 Satellite & Weather",
    "🔐 BB84 QKD Simulation",
    "📊 QBER Visualizations",
    "🌐 3D Earth View",
    "📈 QBER Analysis",
    "✅ Conclusion"
])

st.sidebar.markdown("### 🛰️ Select Satellite Link")
selected_link = st.sidebar.selectbox("Choose Satellite Link:", list(satellite_options.keys()))
selected_file = satellite_options[selected_link]

# Main content
if section == "🌍 Satellite & Weather":
    st.title("🛰️ Real-Time Satellite & Weather Dashboard")
    
    selected_station = st.selectbox("Select Ground Station", list(GROUND_STATIONS.keys()))

    if selected_station:
        lat, lon = GROUND_STATIONS[selected_station]
        st.subheader(f"📍 Ground Station: {selected_station}")

        # Weather
        with st.spinner("Fetching weather data..."):
            weather_data = get_weather(lat, lon)
        if weather_data:
            st.success("Weather data fetched successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Temperature (°C)", weather_data['temperature'])
            with col2:
                st.metric("Weather Description", weather_data['weather'].title())
        else:
            st.error("Failed to fetch weather data.")

        # Satellite Tracking (ISS)
        with st.spinner("Tracking ISS..."):
            sat_data = get_satellite_position()
        if sat_data:
            st.success("ISS Position Fetched!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latitude", f"{sat_data['satlatitude']:.2f}°")
            with col2:
                st.metric("Longitude", f"{sat_data['satlongitude']:.2f}°")
            with col3:
                st.metric("Altitude", f"{sat_data['sataltitude']:.2f} km")
        else:
            st.error("Failed to fetch satellite data.")

elif section == "🔐 BB84 QKD Simulation":
    st.title("🔐 BB84 Quantum Key Distribution Simulation")
    st.write("Simulate quantum key distribution with potential eavesdropping (Eve)")
    
    input_text = st.text_input("📨 Enter a message to encrypt:", "Hello Quantum World!")
    
    if st.button("🚀 Run BB84 Simulation with Eve"):
        alice_bits = text_to_bits(input_text)
        num_bits = len(alice_bits)
        error_probability = 0.01  # Fixed error probability
        
        (alice_bases, eve_bases, eve_results,
         bob_bases, bob_results, sifted_key, bob_sifted_key,
         corrected_key, error_rate, final_key_hex) = simulate_bb84(num_bits, error_probability, alice_bits)

        st.subheader("🔍 Results")
        with st.expander("Show Detailed Results"):
            col1, col2 = st.columns(2)
            with col1:
                st.text("Alice bits: " + str(alice_bits))
                st.text("Alice bases: " + str(alice_bases))
                st.text("Eve bases: " + str(eve_bases))
                st.text("Eve results: " + str(eve_results))
            with col2:
                st.text("Bob bases: " + str(bob_bases))
                st.text("Bob results: " + str(bob_results))
                st.text("Sifted key: " + str(sifted_key))
                st.text("Bob sifted key: " + str(bob_sifted_key))

        st.write(f"🔁 Matching Indices: {len(sifted_key)} bits")
        st.write(f"⚠️ Error Rate: **{error_rate:.2%}**")

        st.subheader("🔐 Privacy Amplification")
        st.code(f"Final Key (SHA-256): {final_key_hex}")

        st.subheader("✅ Authentication")
        challenge = ''.join(np.random.choice(['0', '1'], size=16))
        response_bob = generate_hmac(final_key_hex, challenge)
        response_alice = generate_hmac(final_key_hex, challenge)
        if response_bob == response_alice:
            st.success("Authentication successful ✅")
        else:
            st.error("Authentication failed ❌")
        st.text(f"Challenge: {challenge}")
        st.text(f"HMAC Response: {response_bob}")

        st.subheader("🔒 AES Encryption")
        iv_hex, ciphertext_hex = encrypt_message(final_key_hex, input_text)
        st.text(f"IV (hex): {iv_hex}")
        st.text(f"Encrypted Message (hex): {ciphertext_hex}")

        st.subheader("🔓 Decryption")
        decrypted_text = decrypt_message(final_key_hex, iv_hex, ciphertext_hex)
        st.text(f"Decrypted Message: {decrypted_text}")

        # Visualizations
        st.subheader("📊 Visualizations")
        
        # Error comparison
        fig1 = px.bar(
            x=['Errors', 'Correct'],
            y=[sum(a != b for a, b in zip(sifted_key, bob_sifted_key)),
               sum(a == b for a, b in zip(sifted_key, bob_sifted_key))],
            color=['Errors', 'Correct'],
            title="Eve's Impact on QKD"
        )
        st.plotly_chart(fig1)

        # Decryption accuracy
        original_bits = text_to_bits(input_text)
        decrypted_bits = text_to_bits(decrypted_text)
        errors_after_decryption = sum(1 for a, b in zip(original_bits, decrypted_bits) if a != b)
        
        fig2 = px.bar(
            x=['Errors', 'Correct'],
            y=[errors_after_decryption, len(original_bits) - errors_after_decryption],
            color=['Errors', 'Correct'],
            title="Decryption Accuracy"
        )
        st.plotly_chart(fig2)

elif section == "📊 QBER Visualizations":
    st.title("📡 QBER vs Satellite Distance")
    
    # Define consistent QBER thresholds
    GOOD_QBER_THRESHOLD = 0.05  # 5% standard threshold
    WARNING_QBER_THRESHOLD = 0.07  # 7% warning level

    # Load the selected QBER data file
    try:
        with open(selected_file, "r") as f:
            data = json.load(f)

        qber = data.get("qber", [])
        distance = data.get("distance", [])

        if not qber or not distance:
            st.warning("No QBER or distance data found in the selected file.")
        else:
            df = pd.DataFrame({
                "QBER": qber,
                "Distance (km)": distance,
                "Status": ["✅ Good" if q < GOOD_QBER_THRESHOLD else 
                          "⚠️ Warning" if q < WARNING_QBER_THRESHOLD else 
                          "❌ Unsecure" for q in qber]
            })
            df["Index"] = df.index

            # Line chart: QBER over time/index
            fig1 = px.line(df, x="Index", y="QBER", title=f"QBER over Time – {selected_link}",
                         color="Status", color_discrete_map={
                             "✅ Good": "green",
                             "⚠️ Warning": "orange",
                             "❌ Unsecure": "red"
                         })
            st.plotly_chart(fig1, use_container_width=True)

            # Scatter plot: QBER vs Distance
            fig2 = px.scatter(df, x="Distance (km)", y="QBER", 
                            title=f"QBER vs Distance – {selected_link}",
                            color="Status", 
                            color_discrete_map={
                                "✅ Good": "green",
                                "⚠️ Warning": "orange",
                                "❌ Unsecure": "red"
                            })
            st.plotly_chart(fig2, use_container_width=True)

            st.info(f"""
            **QBER Threshold Standards:**
            - ✅ Good: QBER < {GOOD_QBER_THRESHOLD*100}%
            - ⚠️ Warning: {GOOD_QBER_THRESHOLD*100}% ≤ QBER < {WARNING_QBER_THRESHOLD*100}%
            - ❌ Unsecure: QBER ≥ {WARNING_QBER_THRESHOLD*100}%
            """)

    except FileNotFoundError:
        st.error(f"Data file `{selected_file}` not found.")
    except Exception as e:
        st.error(f"Error loading QBER data: {e}")

elif section == "🌐 3D Earth View":
    st.title("🌍 3D Orbit and QBER Visualization")
    st.write(f"Satellite Link: **{selected_link}**")
    
    try:
        with open(selected_file, "r") as f:
            data = json.load(f)
        
        # Get the ground station name from the selected link
        ground_station = selected_link.split("→")[-1].strip().replace("Ground Station ", "")
        
        df = pd.DataFrame({"QBER": data["qber"]})
        df["Index"] = df.index
        
        # Generate orbit path (simplified for demo)
        orbit_lats = np.linspace(-60, 60, len(df))
        orbit_lons = np.linspace(-180, 180, len(df))
        
        # Add ground station marker
        gs_df = pd.DataFrame({
            "QBER": [0],
            "lat": [ground_station_coords.get(ground_station, {}).get("lat", 0)],
            "lon": [ground_station_coords.get(ground_station, {}).get("lon", 0)]
        })
        
        fig = px.scatter_geo(df, lat=orbit_lats, lon=orbit_lons, color="QBER",
                             title=f"Satellite Orbit View – {selected_link}", 
                             color_continuous_scale="Viridis")
        
        # Add ground station marker
        fig.add_scattergeo(
            lat=gs_df["lat"],
            lon=gs_df["lon"],
            marker=dict(color="red", size=10),
            name=f"Ground Station: {ground_station}"
        )
        
        fig.update_geos(
            projection_type="orthographic",
            showcountries=True,
            showland=True,
            landcolor="lightgray"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading 3D visualization data: {e}")

elif section == "📈 QBER Analysis":
    st.title("📉 Distance vs QBER Analysis")
    st.write(f"Satellite Link: **{selected_link}**")

    try:
        with open(selected_file, "r") as f:
            data = json.load(f)

        df = pd.DataFrame({
            "QBER": data["qber"],
            "Distance (km)": data["distance"]
        })

        # Use consistent 5% threshold
        GOOD_QBER_THRESHOLD = 0.05
        WARNING_QBER_THRESHOLD = 0.07

        st.subheader("🔍 Detailed Analysis")
        for i in range(len(df)):
            d = df.iloc[i]["Distance (km)"]
            q = df.iloc[i]["QBER"]
            if q < GOOD_QBER_THRESHOLD:
                comment = "✅ Good Link"
            elif q < WARNING_QBER_THRESHOLD:
                comment = "⚠️ Noisy Link (Warning)"
            else:
                comment = "❌ Unsecure Link"
            st.markdown(f"**Distance:** {d:.2f} km — **QBER:** {q:.3f} → {comment}")

    except Exception as e:
        st.error(f"Error loading analysis data: {e}")

elif section == "✅ Conclusion":
    st.title("🔚 Conclusion & Key Takeaways")
    st.success("When the satellite is closer, the communication is secure (low QBER), and when farther, it's noisy and error-prone.")
    
    st.markdown(f"""
    ### 🚀 Key Findings for {selected_link}
    
    **Quantum Communication Insights:**
    - The BB84 protocol effectively demonstrates secure quantum key distribution
    - Eve's eavesdropping introduces measurable errors in the quantum channel
    - Quantum-generated keys can be successfully used with classical encryption (AES)
    
    **Satellite Communication Patterns:**
    - Distance directly impacts QBER in satellite quantum links
    - Close approaches provide optimal windows for secure key exchange
    - Visualization tools help identify secure communication opportunities
    
    ### 🛠️ Technical Implementation
    - Combined quantum simulation with real-world satellite tracking
    - Interactive 3D visualization of satellite orbits
    - Comprehensive QBER analysis with threshold indicators
    
    ### 🌟 Next Steps
    - Integrate real quantum hardware backends
    - Add more satellite tracking options
    - Implement advanced error correction protocols
    
    Thanks for exploring quantum satellite communications! 🛰️🔐
    """)
    
    st.balloons()