# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s --M no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:   | llvm-objdump -d --mattr=+zve32f -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vfsqrt.v v8, v4, v0.t
# CHECK-INST: vfsqrt.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x4c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfsqrt.v v8, v4
# CHECK-INST: vfsqrt.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfrsqrt7.v v8, v4, v0.t
# CHECK-INST: vfrsqrt7.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x42,0x4c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfrsqrt7.v v8, v4
# CHECK-INST: vfrsqrt7.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x42,0x4e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfrec7.v v8, v4, v0.t
# CHECK-INST: vfrec7.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x42,0x4c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfrec7.v v8, v4
# CHECK-INST: vfrec7.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x42,0x4e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfclass.v v8, v4, v0.t
# CHECK-INST: vfclass.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x4c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfclass.v v8, v4
# CHECK-INST: vfclass.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfmerge.vfm v8, v4, fa0, v0
# CHECK-INST: vfmerge.vfm v8, v4, fa0, v0
# CHECK-ENCODING: [0x57,0x54,0x45,0x5c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-INST: vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x38]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfslide1up.vf v8, v4, fa0
# CHECK-INST: vfslide1up.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3a]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-INST: vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x3c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfslide1down.vf v8, v4, fa0
# CHECK-INST: vfslide1down.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
