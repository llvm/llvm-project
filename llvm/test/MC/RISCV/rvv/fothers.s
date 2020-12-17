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

vfsqrt.v v8, v4, v0.t
# CHECK-INST: vfsqrt.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x4c]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 40 4c <unknown>

vfsqrt.v v8, v4
# CHECK-INST: vfsqrt.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4e]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 40 4e <unknown>

vfclass.v v8, v4, v0.t
# CHECK-INST: vfclass.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x4c]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 48 4c <unknown>

vfclass.v v8, v4
# CHECK-INST: vfclass.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4e]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 48 4e <unknown>

vfmerge.vfm v8, v4, fa0, v0
# CHECK-INST: vfmerge.vfm v8, v4, fa0, v0
# CHECK-ENCODING: [0x57,0x54,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 5c <unknown>

vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-INST: vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x38]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 38 <unknown>

vfslide1up.vf v8, v4, fa0
# CHECK-INST: vfslide1up.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3a]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 3a <unknown>

vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-INST: vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 3c <unknown>

vfslide1down.vf v8, v4, fa0
# CHECK-INST: vfslide1down.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 3e <unknown>
