# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --M no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v --no-print-imm-hex -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vmerge.vvm v8, v4, v20, v0
# CHECK-INST: vmerge.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x5c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vmerge.vxm v8, v4, a0, v0
# CHECK-INST: vmerge.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x5c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vmerge.vim v8, v4, 15, v0
# CHECK-INST: vmerge.vim v8, v4, 15, v0
# CHECK-ENCODING: [0x57,0xb4,0x47,0x5c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslideup.vx v8, v4, a0, v0.t
# CHECK-INST: vslideup.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x38]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslideup.vx v8, v4, a0
# CHECK-INST: vslideup.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3a]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslideup.vi v8, v4, 31, v0.t
# CHECK-INST: vslideup.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x38]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslideup.vi v8, v4, 31
# CHECK-INST: vslideup.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3a]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslidedown.vx v8, v4, a0, v0.t
# CHECK-INST: vslidedown.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x3c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslidedown.vx v8, v4, a0
# CHECK-INST: vslidedown.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3e]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslidedown.vi v8, v4, 31, v0.t
# CHECK-INST: vslidedown.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslidedown.vi v8, v4, 31
# CHECK-INST: vslidedown.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3e]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslide1up.vx v8, v4, a0, v0.t
# CHECK-INST: vslide1up.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x38]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslide1up.vx v8, v4, a0
# CHECK-INST: vslide1up.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3a]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslide1down.vx v8, v4, a0, v0.t
# CHECK-INST: vslide1down.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x3c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vslide1down.vx v8, v4, a0
# CHECK-INST: vslide1down.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3e]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgather.vv v8, v4, v20, v0.t
# CHECK-INST: vrgather.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x30]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgather.vv v8, v4, v20
# CHECK-INST: vrgather.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x32]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgather.vx v8, v4, a0, v0.t
# CHECK-INST: vrgather.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x30]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgather.vx v8, v4, a0
# CHECK-INST: vrgather.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x32]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgather.vi v8, v4, 31, v0.t
# CHECK-INST: vrgather.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x30]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgather.vi v8, v4, 31
# CHECK-INST: vrgather.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x32]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgatherei16.vv v8, v4, v20, v0.t
# CHECK-INST: vrgatherei16.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x38]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vrgatherei16.vv v8, v4, v20
# CHECK-INST: vrgatherei16.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x3a]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vcompress.vm v8, v4, v20
# CHECK-INST: vcompress.vm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x5e]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
