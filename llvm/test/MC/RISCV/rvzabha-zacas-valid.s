# RUN: llvm-mc %s -triple=riscv32 -mattr=+a,+zabha,+zacas -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a,+zabha,+zacas -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a,+zabha,+zacas < %s \
# RUN:     | llvm-objdump --mattr=+a,+zabha,+zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a,+zabha,+zacas < %s \
# RUN:     | llvm-objdump --mattr=+a,+zabha,+zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+a,+zabha -show-encoding %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+a,+zabha -show-encoding %s

# CHECK-ASM-AND-OBJ: amocas.b a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0x85,0xd7,0x28]
amocas.b a1, a3, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.b a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0x85,0xd7,0x28]
amocas.b a1, a3, 0(a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.b zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0x80,0x07,0x28]
amocas.b zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.b.aq zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0x80,0x07,0x2c]
amocas.b.aq zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.b.rl zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0x80,0x07,0x2a]
amocas.b.rl zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.b.aqrl zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0x80,0x07,0x2e]
amocas.b.aqrl zero, zero, (a5) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}

# CHECK-ASM-AND-OBJ: amocas.h a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0x95,0xc5,0x28]
amocas.h a0, a2, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.h a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0x95,0xc5,0x28]
amocas.h a0, a2, 0(a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.h zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0x90,0x05,0x28]
amocas.h zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.h.aq zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0x90,0x05,0x2c]
amocas.h.aq zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.h.rl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0x90,0x05,0x2a]
amocas.h.rl zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
# CHECK-ASM-AND-OBJ: amocas.h.aqrl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0x90,0x05,0x2e]
amocas.h.aqrl zero, zero, (a1) # expected-error {{instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions)}}
