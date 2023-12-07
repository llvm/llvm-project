# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zacas -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zacas -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zacas < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zacas < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zacas -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+a -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -mattr=+a -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-ASM-AND-OBJ: amocas.w a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xa5,0xd7,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.w a1, a3, (a5)
# CHECK-ASM-AND-OBJ: amocas.w a1, a3, (a5)
# CHECK-ASM: encoding: [0xaf,0xa5,0xd7,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.w a1, a3, 0(a5)
# CHECK-ASM-AND-OBJ: amocas.w zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.w zero, zero, (a5)
# CHECK-ASM-AND-OBJ: amocas.w.aq zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.w.aq zero, zero, (a5)
# CHECK-ASM-AND-OBJ: amocas.w.rl zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.w.rl zero, zero, (a5)
# CHECK-ASM-AND-OBJ: amocas.w.aqrl zero, zero, (a5)
# CHECK-ASM: encoding: [0x2f,0xa0,0x07,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.w.aqrl zero, zero, (a5)

# CHECK-ASM-AND-OBJ: amocas.d a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0xb5,0xc5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d a0, a2, (a1)
# CHECK-ASM-AND-OBJ: amocas.d a0, a2, (a1)
# CHECK-ASM: encoding: [0x2f,0xb5,0xc5,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d a0, a2, 0(a1)
# CHECK-ASM-AND-OBJ: amocas.d zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x28]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d zero, zero, (a1)
# CHECK-ASM-AND-OBJ: amocas.d.aq zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x2c]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d.aq zero, zero, (a1)
# CHECK-ASM-AND-OBJ: amocas.d.rl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x2a]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d.rl zero, zero, (a1)
# CHECK-ASM-AND-OBJ: amocas.d.aqrl zero, zero, (a1)
# CHECK-ASM: encoding: [0x2f,0xb0,0x05,0x2e]
# CHECK-ERROR: instruction requires the following: 'Zacas' (Atomic Compare-And-Swap Instructions){{$}}
amocas.d.aqrl zero, zero, (a1)
