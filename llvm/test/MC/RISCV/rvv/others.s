# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --riscv-no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v --no-print-imm-hex -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vmerge.vvm v8, v4, v20, v0
# CHECK-INST: vmerge.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x5c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5c4a0457 <unknown>

vmerge.vxm v8, v4, a0, v0
# CHECK-INST: vmerge.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5c454457 <unknown>

vmerge.vim v8, v4, 15, v0
# CHECK-INST: vmerge.vim v8, v4, 15, v0
# CHECK-ENCODING: [0x57,0xb4,0x47,0x5c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5c47b457 <unknown>

vslideup.vx v8, v4, a0, v0.t
# CHECK-INST: vslideup.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x38]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 38454457 <unknown>

vslideup.vx v8, v4, a0
# CHECK-INST: vslideup.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3a454457 <unknown>

vslideup.vi v8, v4, 31, v0.t
# CHECK-INST: vslideup.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x38]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 384fb457 <unknown>

vslideup.vi v8, v4, 31
# CHECK-INST: vslideup.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3a4fb457 <unknown>

vslidedown.vx v8, v4, a0, v0.t
# CHECK-INST: vslidedown.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3c454457 <unknown>

vslidedown.vx v8, v4, a0
# CHECK-INST: vslidedown.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3e454457 <unknown>

vslidedown.vi v8, v4, 31, v0.t
# CHECK-INST: vslidedown.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3c4fb457 <unknown>

vslidedown.vi v8, v4, 31
# CHECK-INST: vslidedown.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3e4fb457 <unknown>

vslide1up.vx v8, v4, a0, v0.t
# CHECK-INST: vslide1up.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x38]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 38456457 <unknown>

vslide1up.vx v8, v4, a0
# CHECK-INST: vslide1up.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3a456457 <unknown>

vslide1down.vx v8, v4, a0, v0.t
# CHECK-INST: vslide1down.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3c456457 <unknown>

vslide1down.vx v8, v4, a0
# CHECK-INST: vslide1down.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3e456457 <unknown>

vrgather.vv v8, v4, v20, v0.t
# CHECK-INST: vrgather.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x30]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 304a0457 <unknown>

vrgather.vv v8, v4, v20
# CHECK-INST: vrgather.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x32]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 324a0457 <unknown>

vrgather.vx v8, v4, a0, v0.t
# CHECK-INST: vrgather.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x30]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 30454457 <unknown>

vrgather.vx v8, v4, a0
# CHECK-INST: vrgather.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x32]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 32454457 <unknown>

vrgather.vi v8, v4, 31, v0.t
# CHECK-INST: vrgather.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x30]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 304fb457 <unknown>

vrgather.vi v8, v4, 31
# CHECK-INST: vrgather.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x32]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 324fb457 <unknown>

vrgatherei16.vv v8, v4, v20, v0.t
# CHECK-INST: vrgatherei16.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x38]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 384a0457 <unknown>

vrgatherei16.vv v8, v4, v20
# CHECK-INST: vrgatherei16.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x3a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3a4a0457 <unknown>

vcompress.vm v8, v4, v20
# CHECK-INST: vcompress.vm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5e4a2457 <unknown>
