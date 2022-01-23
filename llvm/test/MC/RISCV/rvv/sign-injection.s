# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfsgnj.vv v8, v4, v20, v0.t
# CHECK-INST: vfsgnj.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x20]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 14 4a 20 <unknown>

vfsgnj.vv v8, v4, v20
# CHECK-INST: vfsgnj.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x22]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 14 4a 22 <unknown>

vfsgnj.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsgnj.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 54 45 20 <unknown>

vfsgnj.vf v8, v4, fa0
# CHECK-INST: vfsgnj.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 54 45 22 <unknown>

vfsgnjn.vv v8, v4, v20, v0.t
# CHECK-INST: vfsgnjn.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x24]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 14 4a 24 <unknown>

vfsgnjn.vv v8, v4, v20
# CHECK-INST: vfsgnjn.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x26]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 14 4a 26 <unknown>

vfsgnjn.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsgnjn.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 54 45 24 <unknown>

vfsgnjn.vf v8, v4, fa0
# CHECK-INST: vfsgnjn.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 54 45 26 <unknown>

vfsgnjx.vv v8, v4, v20, v0.t
# CHECK-INST: vfsgnjx.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x28]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 14 4a 28 <unknown>

vfsgnjx.vv v8, v4, v20
# CHECK-INST: vfsgnjx.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x2a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 14 4a 2a <unknown>

vfsgnjx.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsgnjx.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x28]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 54 45 28 <unknown>

vfsgnjx.vf v8, v4, fa0
# CHECK-INST: vfsgnjx.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x2a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f', 'Zve64f' or 'Zve64d'
# CHECK-UNKNOWN: 57 54 45 2a <unknown>
