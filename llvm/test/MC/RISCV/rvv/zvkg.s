# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvkg %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvkg %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+experimental-zvkg  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvkg %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vghsh.vv v10, v9, v8
# CHECK-INST: vghsh.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvkg' (Vector GCM instructions for Cryptography.){{$}}
# CHECK-UNKNOWN: 77 25 94 b2   <unknown>
