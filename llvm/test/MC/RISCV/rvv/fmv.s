# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfmv.v.f v8, fa0
# CHECK-INST: vfmv.v.f v8, fa0
# CHECK-ENCODING: [0x57,0x54,0x05,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 54 05 5e <unknown>

vfmv.f.s fa0, v4
# CHECK-INST: vfmv.f.s fa0, v4
# CHECK-ENCODING: [0x57,0x15,0x40,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 15 40 42 <unknown>

vfmv.s.f v8, fa0
# CHECK-INST: vfmv.s.f v8, fa0
# CHECK-ENCODING: [0x57,0x54,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 54 05 42 <unknown>
