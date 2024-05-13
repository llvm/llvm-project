# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsll.vv v8, v4, v20, v0.t
# CHECK-INST: vsll.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 944a0457 <unknown>

vsll.vv v8, v4, v20
# CHECK-INST: vsll.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 964a0457 <unknown>

vsll.vx v8, v4, a0, v0.t
# CHECK-INST: vsll.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 94454457 <unknown>

vsll.vx v8, v4, a0
# CHECK-INST: vsll.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 96454457 <unknown>

vsll.vi v8, v4, 31, v0.t
# CHECK-INST: vsll.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 944fb457 <unknown>

vsll.vi v8, v4, 31
# CHECK-INST: vsll.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 964fb457 <unknown>

vsrl.vv v8, v4, v20, v0.t
# CHECK-INST: vsrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a04a0457 <unknown>

vsrl.vv v8, v4, v20
# CHECK-INST: vsrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a24a0457 <unknown>

vsrl.vx v8, v4, a0, v0.t
# CHECK-INST: vsrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0454457 <unknown>

vsrl.vx v8, v4, a0
# CHECK-INST: vsrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2454457 <unknown>

vsrl.vi v8, v4, 31, v0.t
# CHECK-INST: vsrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a04fb457 <unknown>

vsrl.vi v8, v4, 31
# CHECK-INST: vsrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a24fb457 <unknown>

vsra.vv v8, v4, v20, v0.t
# CHECK-INST: vsra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a44a0457 <unknown>

vsra.vv v8, v4, v20
# CHECK-INST: vsra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a64a0457 <unknown>

vsra.vx v8, v4, a0, v0.t
# CHECK-INST: vsra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4454457 <unknown>

vsra.vx v8, v4, a0
# CHECK-INST: vsra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6454457 <unknown>

vsra.vi v8, v4, 31, v0.t
# CHECK-INST: vsra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a44fb457 <unknown>

vsra.vi v8, v4, 31
# CHECK-INST: vsra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a64fb457 <unknown>

vnsrl.wv v8, v4, v20, v0.t
# CHECK-INST: vnsrl.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b04a0457 <unknown>

vnsrl.wv v4, v4, v20, v0.t
# CHECK-INST: vnsrl.wv v4, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x02,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b04a0257 <unknown>

vnsrl.wv v8, v4, v20
# CHECK-INST: vnsrl.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b24a0457 <unknown>

vnsrl.wx v8, v4, a0, v0.t
# CHECK-INST: vnsrl.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b0454457 <unknown>

vnsrl.wx v8, v4, a0
# CHECK-INST: vnsrl.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b2454457 <unknown>

vnsrl.wi v8, v4, 31, v0.t
# CHECK-INST: vnsrl.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b04fb457 <unknown>

vnsrl.wi v8, v4, 31
# CHECK-INST: vnsrl.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b24fb457 <unknown>

vnsra.wv v8, v4, v20, v0.t
# CHECK-INST: vnsra.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b44a0457 <unknown>

vnsra.wv v8, v4, v20
# CHECK-INST: vnsra.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b64a0457 <unknown>

vnsra.wx v8, v4, a0, v0.t
# CHECK-INST: vnsra.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b4454457 <unknown>

vnsra.wx v8, v4, a0
# CHECK-INST: vnsra.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b6454457 <unknown>

vnsra.wi v8, v4, 31, v0.t
# CHECK-INST: vnsra.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b44fb457 <unknown>

vnsra.wi v8, v4, 31
# CHECK-INST: vnsra.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b64fb457 <unknown>

vssrl.vv v8, v4, v20, v0.t
# CHECK-INST: vssrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a84a0457 <unknown>

vssrl.vv v8, v4, v20
# CHECK-INST: vssrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aa4a0457 <unknown>

vssrl.vx v8, v4, a0, v0.t
# CHECK-INST: vssrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8454457 <unknown>

vssrl.vx v8, v4, a0
# CHECK-INST: vssrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aa454457 <unknown>

vssrl.vi v8, v4, 31, v0.t
# CHECK-INST: vssrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a84fb457 <unknown>

vssrl.vi v8, v4, 31
# CHECK-INST: vssrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aa4fb457 <unknown>

vssra.vv v8, v4, v20, v0.t
# CHECK-INST: vssra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac4a0457 <unknown>

vssra.vv v8, v4, v20
# CHECK-INST: vssra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae4a0457 <unknown>

vssra.vx v8, v4, a0, v0.t
# CHECK-INST: vssra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac454457 <unknown>

vssra.vx v8, v4, a0
# CHECK-INST: vssra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae454457 <unknown>

vssra.vi v8, v4, 31, v0.t
# CHECK-INST: vssra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac4fb457 <unknown>

vssra.vi v8, v4, 31
# CHECK-INST: vssra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae4fb457 <unknown>
