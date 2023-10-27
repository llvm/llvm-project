# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve64x --mattr=+experimental-zvbc %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve64x --mattr=+experimental-zvbc %s \
# RUN:        | llvm-objdump -d --mattr=+zve64x --mattr=+experimental-zvbc  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve64x --mattr=+experimental-zvbc %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vclmul.vv v10, v9, v8
# CHECK-INST: vclmul.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvbc' (Vector Carryless Multiplication){{$}}
# CHECK-UNKNOWN: 57 25 94 32   <unknown>

vclmul.vx v10, v9, a0
# CHECK-INST: vclmul.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x32]
# CHECK-ERROR: instruction requires the following: 'Zvbc' (Vector Carryless Multiplication){{$}}
# CHECK-UNKNOWN: 57 65 95 32   <unknown>

vclmulh.vv v10, v9, v8
# CHECK-INST: vclmulh.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x36]
# CHECK-ERROR: instruction requires the following: 'Zvbc' (Vector Carryless Multiplication){{$}}
# CHECK-UNKNOWN: 57 25 94 36   <unknown>

vclmulh.vx v10, v9, a0
# CHECK-INST: vclmulh.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x36]
# CHECK-ERROR: instruction requires the following: 'Zvbc' (Vector Carryless Multiplication){{$}}
# CHECK-UNKNOWN: 57 65 95 36   <unknown>
