import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portsList = []

print("Porte disponibili:")
for onePort in ports:
    portsList.append(onePort.device)
    print(onePort.device)

val = input("Seleziona la porta (es. COM4): ").strip()

if val in portsList:
    portVar = val
    print(f"Porta selezionata: {portVar}")
else:
    print("Porta non trovata.")
    exit()

serialInst.baudrate = 9600
serialInst.port = portVar

try:
    serialInst.open()
    print(f"Porta {portVar} aperta con successo.")
except Exception as e:
    print(f"Errore nell'apertura della porta {portVar}: {e}")
    exit()

while True:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        print(packet.decode('utf').rstrip('\n'))
