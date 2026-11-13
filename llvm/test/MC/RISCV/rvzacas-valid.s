# RUN: llvm-mc %s -triple=riscv32 -mattr=+a,+zacas -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a,+zacas -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a,+zacas < %s \
# RUN:     | llvm-objdump --mattr=+a,+zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a,+zacas < %s \
# RUN:     | llvm-objdump --mattr=+a,+zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+a -show-encoding %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+a -show-encoding %s

# CHECK-ASM-AND-OBJ: amocas.w a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xa5,0xd7,0x28]
amocas.w a1, a3, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.w a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xa5,0xd7,0x28]
amocas.w a1, a3, 0(a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.w zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x28]
amocas.w zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.w.aq zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x2c]
amocas.w.aq zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.w.rl zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x2a]
amocas.w.rl zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.w.aqrl zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x2e]
amocas.w.aqrl zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}

# CHECK-ASM-AND-OBJ: amocas.d a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0xb5,0xc5,0x28]
amocas.d a0, a2, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.d a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0xb5,0xc5,0x28]
amocas.d a0, a2, 0(a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.d zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x28]
amocas.d zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.d.aq zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x2c]
amocas.d.aq zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.d.rl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x2a]
amocas.d.rl zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.d.aqrl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x2e]
amocas.d.aqrl zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
