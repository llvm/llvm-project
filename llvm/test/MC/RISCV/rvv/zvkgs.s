# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvkgs %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvkgs %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+experimental-zvkgs  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvkgs %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vghsh.vs v10, v9, v8
# CHECK-INST: vghsh.vs v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0x8e]
# CHECK-ERROR: instruction requires the following: 'Zvkgs' (Vector-Scalar GCM instructions for Cryptography){{$}}
# CHECK-UNKNOWN: 8e942577 <unknown>

vgmul.vs v10, v9
# CHECK-INST: vgmul.vs v10, v9
# CHECK-ENCODING: [0x77,0xa5,0x98,0xa6]
# CHECK-ERROR: instruction requires the following: 'Zvkgs' (Vector-Scalar GCM instructions for Cryptography){{$}}
# CHECK-UNKNOWN: a698a577 <unknown>
