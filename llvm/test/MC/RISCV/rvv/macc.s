# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmacc.vv v8, v20, v4
# CHECK-INST: vmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmacc.vx v8, a0, v4, v0.t
# CHECK-INST: vmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmacc.vx v8, a0, v4
# CHECK-INST: vmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xbc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsac.vv v8, v20, v4
# CHECK-INST: vnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xbe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsac.vx v8, a0, v4, v0.t
# CHECK-INST: vnmsac.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsac.vx v8, a0, v4
# CHECK-INST: vnmsac.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadd.vv v8, v20, v4, v0.t
# CHECK-INST: vmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadd.vv v8, v20, v4
# CHECK-INST: vmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadd.vx v8, a0, v4, v0.t
# CHECK-INST: vmadd.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadd.vx v8, a0, v4
# CHECK-INST: vmadd.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsub.vv v8, v20, v4, v0.t
# CHECK-INST: vnmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsub.vv v8, v20, v4
# CHECK-INST: vnmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsub.vx v8, a0, v4, v0.t
# CHECK-INST: vnmsub.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vnmsub.vx v8, a0, v4
# CHECK-INST: vnmsub.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccu.vv v8, v20, v4, v0.t
# CHECK-INST: vwmaccu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccu.vv v8, v20, v4
# CHECK-INST: vwmaccu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccu.vx v8, a0, v4, v0.t
# CHECK-INST: vwmaccu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccu.vx v8, a0, v4
# CHECK-INST: vwmaccu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xf2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vwmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmacc.vv v8, v20, v4
# CHECK-INST: vwmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmacc.vx v8, a0, v4, v0.t
# CHECK-INST: vwmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmacc.vx v8, a0, v4
# CHECK-INST: vwmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xf6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccsu.vv v8, v20, v4, v0.t
# CHECK-INST: vwmaccsu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xfc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccsu.vv v8, v20, v4
# CHECK-INST: vwmaccsu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xfe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccsu.vx v8, a0, v4, v0.t
# CHECK-INST: vwmaccsu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccsu.vx v8, a0, v4
# CHECK-INST: vwmaccsu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccus.vx v8, a0, v4, v0.t
# CHECK-INST: vwmaccus.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmaccus.vx v8, a0, v4
# CHECK-INST: vwmaccus.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xfa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
