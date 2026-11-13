# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vfsgnj.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnj.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x20]

vfsgnj.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnj.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x22]

vfsgnj.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnj.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x20]

vfsgnj.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnj.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x22]

vfsgnjn.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjn.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x24]

vfsgnjn.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjn.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x26]

vfsgnjn.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjn.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x24]

vfsgnjn.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjn.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x26]

vfsgnjx.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjx.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x28]

vfsgnjx.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjx.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x2a]

vfsgnjx.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjx.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x28]

vfsgnjx.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsgnjx.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x2a]
