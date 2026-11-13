# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsll.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsll.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x94]

vsll.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsll.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x96]

vsll.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsll.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x94]

vsll.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsll.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x96]

vsll.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsll.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x94]

vsll.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsll.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x96]

vsrl.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa0]

vsrl.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa2]

vsrl.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa0]

vsrl.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa2]

vsrl.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa0]

vsrl.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa2]

vsra.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa4]

vsra.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa6]

vsra.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa4]

vsra.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa6]

vsra.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa4]

vsra.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa6]

vnsrl.wv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb0]

vnsrl.wv v4, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wv v4, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x02,0x4a,0xb0]

vnsrl.wv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb2]

vnsrl.wx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb0]

vnsrl.wx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb2]

vnsrl.wi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb0]

vnsrl.wi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsrl.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb2]

vnsra.wv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsra.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb4]

vnsra.wv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsra.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb6]

vnsra.wx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsra.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb4]

vnsra.wx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsra.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb6]

vnsra.wi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsra.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb4]

vnsra.wi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vnsra.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb6]

vssrl.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa8]

vssrl.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xaa]

vssrl.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa8]

vssrl.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xaa]

vssrl.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa8]

vssrl.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xaa]

vssra.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xac]

vssra.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xae]

vssra.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xac]

vssra.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xae]

vssra.vi v8, v4, 31, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xac]

vssra.vi v8, v4, 31 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xae]
