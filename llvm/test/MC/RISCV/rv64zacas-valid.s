# RUN: llvm-mc %s -triple=riscv64 -mattr=+a,zacas -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a,zacas < %s \
# RUN:     | llvm-objdump --mattr=+a,zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+a -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# Odd register numbers for rd and rs2 are allowed for amocas.d on RV64.

# CHECK-ASM-AND-OBJ: amocas.d a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xb5,0xd7,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d a1, a3, (a5)
# CHECK-ASM-AND-OBJ: amocas.d.aq a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xb5,0xd7,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d.aq a1, a3, (a5)
# CHECK-ASM-AND-OBJ: amocas.d.rl a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xb5,0xd7,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d.rl a1, a3, (a5)
# CHECK-ASM-AND-OBJ: amocas.d.aqrl a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xb5,0xd7,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d.aqrl a1, a3, (a5)

# CHECK-ASM-AND-OBJ: amocas.q a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0xc5,0xc5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.q a0, a2, (a1)
# CHECK-ASM-AND-OBJ: amocas.q a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0xc5,0xc5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.q a0, a2, 0(a1)
# CHECK-ASM-AND-OBJ: amocas.q zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xc0,0x05,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.q zero, zero, (a1)
# CHECK-ASM-AND-OBJ: amocas.q.aq zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xc0,0x05,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.q.aq zero, zero, (a1)
# CHECK-ASM-AND-OBJ: amocas.q.rl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xc0,0x05,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.q.rl zero, zero, (a1)
# CHECK-ASM-AND-OBJ: amocas.q.aqrl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xc0,0x05,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.q.aqrl zero, zero, (a1)
