# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --M no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v --no-print-imm-hex -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vmerge.vvm v8, v4, v20, v0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmerge.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x5c]

vmerge.vxm v8, v4, a0, v0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmerge.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x5c]

vmerge.vim v8, v4, 15, v0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmerge.vim v8, v4, 15, v0
# CHECK-ENCODING: [0x57,0xb4,0x47,0x5c]

vslideup.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslideup.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x38]

vslideup.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslideup.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3a]

vslideup.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslideup.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x38]

vslideup.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslideup.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3a]

vslidedown.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslidedown.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x3c]

vslidedown.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslidedown.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3e]

vslidedown.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslidedown.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3c]

vslidedown.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslidedown.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3e]

vslide1up.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslide1up.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x38]

vslide1up.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslide1up.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3a]

vslide1down.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslide1down.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x3c]

vslide1down.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vslide1down.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3e]

vrgather.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgather.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x30]

vrgather.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgather.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x32]

vrgather.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgather.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x30]

vrgather.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgather.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x32]

vrgather.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgather.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x30]

vrgather.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgather.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x32]

vrgatherei16.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgatherei16.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x38]

vrgatherei16.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vrgatherei16.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x3a]

vcompress.vm v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vcompress.vm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x5e]
