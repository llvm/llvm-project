# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmacc.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb4]

vmacc.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb6]

vmacc.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xb4]

vmacc.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xb6]

vnmsac.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xbc]

vnmsac.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xbe]

vnmsac.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsac.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xbc]

vnmsac.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsac.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xbe]

vmadd.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa4]

vmadd.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa6]

vmadd.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmadd.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xa4]

vmadd.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmadd.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xa6]

vnmsub.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xac]

vnmsub.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xae]

vnmsub.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsub.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xac]

vnmsub.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnmsub.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xae]

vwmaccu.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf0]

vwmaccu.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf2]

vwmaccu.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf0]

vwmaccu.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xf2]

vwmacc.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf4]

vwmacc.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf6]

vwmacc.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf4]

vwmacc.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xf6]

vwmaccsu.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccsu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xfc]

vwmaccsu.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccsu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xfe]

vwmaccsu.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccsu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xfc]

vwmaccsu.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccsu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xfe]

vwmaccus.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccus.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf8]

vwmaccus.vx v8, a0, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwmaccus.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xfa]
