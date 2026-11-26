# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve64x --mattr=+zvbc %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve64x --mattr=+zvbc %s \
# RUN:        | llvm-objdump -d --mattr=+zve64x --mattr=+zvbc  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvbc32e %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvbc32e %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+experimental-zvbc32e  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vclmul.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvbc' or 'Zvbc32e' (Vector Carryless Multiplication)}}
# CHECK-INST: vclmul.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x32]

vclmul.vx v10, v9, a0 # expected-error {{instruction requires the following: 'Zvbc' or 'Zvbc32e' (Vector Carryless Multiplication)}}
# CHECK-INST: vclmul.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x32]

vclmulh.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvbc' or 'Zvbc32e' (Vector Carryless Multiplication)}}
# CHECK-INST: vclmulh.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x36]

vclmulh.vx v10, v9, a0 # expected-error {{instruction requires the following: 'Zvbc' or 'Zvbc32e' (Vector Carryless Multiplication)}}
# CHECK-INST: vclmulh.vx v10, v9, a0
# CHECK-ENCODING: [0x57,0x65,0x95,0x36]
