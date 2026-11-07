# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vfsub.vv v8, v4, v20, v0.t
# CHECK-INST: vfsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x08]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfsub.vv v8, v4, v20
# CHECK-INST: vfsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0a]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x08]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfsub.vf v8, v4, fa0
# CHECK-INST: vfsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x0a]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfrsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfrsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x9c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfrsub.vf v8, v4, fa0
# CHECK-INST: vfrsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x9e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.vv v8, v4, v20, v0.t
# CHECK-INST: vfwsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc8]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.vv v8, v4, v20
# CHECK-INST: vfwsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xca]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfwsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xc8]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.vf v8, v4, fa0
# CHECK-INST: vfwsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xca]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.wv v8, v4, v20, v0.t
# CHECK-INST: vfwsub.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xd8]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.wv v8, v4, v20
# CHECK-INST: vfwsub.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xda]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.wf v8, v4, fa0, v0.t
# CHECK-INST: vfwsub.wf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xd8]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwsub.wf v8, v4, fa0
# CHECK-INST: vfwsub.wf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xda]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
