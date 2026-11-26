# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvknha %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve64x --mattr=+zvknhb %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvknha %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvknha  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve64x --mattr=+zvknhb %s \
# RUN:        | llvm-objdump -d --mattr=+zve64x --mattr=+zvknhb  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s

vsha2ms.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvknha' (Vector SHA-2 (SHA-256 only))}}
# CHECK-INST: vsha2ms.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xb6]

vsha2ch.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvknha' (Vector SHA-2 (SHA-256 only))}}
# CHECK-INST: vsha2ch.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xba]

vsha2cl.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvknha' (Vector SHA-2 (SHA-256 only))}}
# CHECK-INST: vsha2cl.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xbe]
