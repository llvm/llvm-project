# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvkg %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvkg %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvkg  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vghsh.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvkg' (Vector GCM instructions for Cryptography)}}
# CHECK-INST: vghsh.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xb2]

vgmul.vv v10, v9 # expected-error {{instruction requires the following: 'Zvkg' (Vector GCM instructions for Cryptography)}}
# CHECK-INST: vgmul.vv v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x98,0xa2]
