# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+experimental-v --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfmin.vv v8, v4, v20, v0.t
# CHECK-INST: vfmin.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x10]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a 10 <unknown>

vfmin.vv v8, v4, v20
# CHECK-INST: vfmin.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x12]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a 12 <unknown>

vfmin.vf v8, v4, fa0, v0.t
# CHECK-INST: vfmin.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x10]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 10 <unknown>

vfmin.vf v8, v4, fa0
# CHECK-INST: vfmin.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x12]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 12 <unknown>

vfmax.vv v8, v4, v20, v0.t
# CHECK-INST: vfmax.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x18]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a 18 <unknown>

vfmax.vv v8, v4, v20
# CHECK-INST: vfmax.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1a]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a 1a <unknown>

vfmax.vf v8, v4, fa0, v0.t
# CHECK-INST: vfmax.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x18]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 18 <unknown>

vfmax.vf v8, v4, fa0
# CHECK-INST: vfmax.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x1a]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 1a <unknown>
