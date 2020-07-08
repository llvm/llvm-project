# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfdiv.vv v8, v4, v20, v0.t
# CHECK-INST: vfdiv.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 4a 80 <unknown>

vfdiv.vv v8, v4, v20
# CHECK-INST: vfdiv.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 4a 82 <unknown>

vfdiv.vf v8, v4, fa0, v0.t
# CHECK-INST: vfdiv.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 54 45 80 <unknown>

vfdiv.vf v8, v4, fa0
# CHECK-INST: vfdiv.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 54 45 82 <unknown>

vfrdiv.vf v8, v4, fa0, v0.t
# CHECK-INST: vfrdiv.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 54 45 84 <unknown>

vfrdiv.vf v8, v4, fa0
# CHECK-INST: vfrdiv.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 54 45 86 <unknown>
