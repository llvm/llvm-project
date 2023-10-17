# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvknha %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve64x --mattr=+experimental-zvknhb %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvknha %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+experimental-zvknha  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve64x --mattr=+experimental-zvknhb %s \
# RUN:        | llvm-objdump -d --mattr=+zve64x --mattr=+experimental-zvknhb  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvknha %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve64x --mattr=+experimental-zvknhb %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsha2ms.vv v10, v9, v8
# CHECK-INST: vsha2ms.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xb6]
# CHECK-UNKNOWN: 77 25 94 b6   <unknown>
# CHECK-ERROR: instruction requires the following: 'Zvknha' (Vector SHA-2 (SHA-256 only)){{$}}

vsha2ch.vv v10, v9, v8
# CHECK-INST: vsha2ch.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xba]
# CHECK-UNKNOWN: 77 25 94 ba   <unknown>
# CHECK-ERROR: instruction requires the following: 'Zvknha' (Vector SHA-2 (SHA-256 only)){{$}}

vsha2cl.vv v10, v9, v8
# CHECK-INST: vsha2cl.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xbe]
# CHECK-UNKNOWN: 77 25 94 be   <unknown>
# CHECK-ERROR: instruction requires the following: 'Zvknha' (Vector SHA-2 (SHA-256 only)){{$}}
