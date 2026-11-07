# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vfmul.vv v8, v4, v20, v0.t
# CHECK-INST: vfmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x90]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfmul.vv v8, v4, v20
# CHECK-INST: vfmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x92]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfmul.vf v8, v4, fa0, v0.t
# CHECK-INST: vfmul.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x90]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfmul.vf v8, v4, fa0
# CHECK-INST: vfmul.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x92]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwmul.vv v8, v4, v20, v0.t
# CHECK-INST: vfwmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xe0]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwmul.vv v8, v4, v20
# CHECK-INST: vfwmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xe2]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwmul.vf v8, v4, fa0, v0.t
# CHECK-INST: vfwmul.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xe0]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwmul.vf v8, v4, fa0
# CHECK-INST: vfwmul.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xe2]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
