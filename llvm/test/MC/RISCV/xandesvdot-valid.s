# XAndesVDot - Andes Vector Dot Product Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvdot -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvdot < %s \
# RUN:     | llvm-objdump --mattr=+xandesvdot -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvdot -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvdot < %s \
# RUN:     | llvm-objdump --mattr=+xandesvdot -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-OBJ: nds.vd4dots.vv v8, v10, v12
# CHECK-ASM: nds.vd4dots.vv v8, v10, v12
# CHECK-ASM: encoding: [0x5b,0x44,0xc5,0x12]
# CHECK-ERROR: instruction requires the following: 'XAndesVDot' (Andes Vector Dot Product Extension){{$}}
nds.vd4dots.vv v8, v10, v12

# CHECK-OBJ: nds.vd4dots.vv v8, v10, v12, v0.t
# CHECK-ASM: nds.vd4dots.vv v8, v10, v12, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xc5,0x10]
# CHECK-ERROR: instruction requires the following: 'XAndesVDot' (Andes Vector Dot Product Extension){{$}}
nds.vd4dots.vv v8, v10, v12, v0.t

# CHECK-OBJ: nds.vd4dotu.vv v8, v10, v12
# CHECK-ASM: nds.vd4dotu.vv v8, v10, v12
# CHECK-ASM: encoding: [0x5b,0x44,0xc5,0x1e]
# CHECK-ERROR: instruction requires the following: 'XAndesVDot' (Andes Vector Dot Product Extension){{$}}
nds.vd4dotu.vv v8, v10, v12

# CHECK-OBJ: nds.vd4dotu.vv v8, v10, v12, v0.t
# CHECK-ASM: nds.vd4dotu.vv v8, v10, v12, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xc5,0x1c]
# CHECK-ERROR: instruction requires the following: 'XAndesVDot' (Andes Vector Dot Product Extension){{$}}
nds.vd4dotu.vv v8, v10, v12, v0.t

# CHECK-OBJ: nds.vd4dotsu.vv v8, v10, v12
# CHECK-ASM: nds.vd4dotsu.vv v8, v10, v12
# CHECK-ASM: encoding: [0x5b,0x44,0xc5,0x16]
# CHECK-ERROR: instruction requires the following: 'XAndesVDot' (Andes Vector Dot Product Extension){{$}}
nds.vd4dotsu.vv v8, v10, v12

# CHECK-OBJ: nds.vd4dotsu.vv v8, v10, v12, v0.t
# CHECK-ASM: nds.vd4dotsu.vv v8, v10, v12, v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0xc5,0x14]
# CHECK-ERROR: instruction requires the following: 'XAndesVDot' (Andes Vector Dot Product Extension){{$}}
nds.vd4dotsu.vv v8, v10, v12, v0.t
