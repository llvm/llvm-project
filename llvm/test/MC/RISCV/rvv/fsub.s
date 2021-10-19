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

vfsub.vv v8, v4, v20, v0.t
# CHECK-INST: vfsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a 08 <unknown>

vfsub.vv v8, v4, v20
# CHECK-INST: vfsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a 0a <unknown>

vfsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 08 <unknown>

vfsub.vf v8, v4, fa0
# CHECK-INST: vfsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 0a <unknown>

vfrsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfrsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 9c <unknown>

vfrsub.vf v8, v4, fa0
# CHECK-INST: vfrsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 9e <unknown>

vfwsub.vv v8, v4, v20, v0.t
# CHECK-INST: vfwsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc8]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a c8 <unknown>

vfwsub.vv v8, v4, v20
# CHECK-INST: vfwsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xca]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a ca <unknown>

vfwsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfwsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xc8]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 c8 <unknown>

vfwsub.vf v8, v4, fa0
# CHECK-INST: vfwsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xca]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 ca <unknown>

vfwsub.wv v8, v4, v20, v0.t
# CHECK-INST: vfwsub.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xd8]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a d8 <unknown>

vfwsub.wv v8, v4, v20
# CHECK-INST: vfwsub.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xda]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 14 4a da <unknown>

vfwsub.wf v8, v4, fa0, v0.t
# CHECK-INST: vfwsub.wf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xd8]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 d8 <unknown>

vfwsub.wf v8, v4, fa0
# CHECK-INST: vfwsub.wf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xda]
# CHECK-ERROR: instruction requires the following: 'F'{{.*}}'V'
# CHECK-UNKNOWN: 57 54 45 da <unknown>
