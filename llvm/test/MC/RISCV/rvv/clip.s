# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vnclipu.wv v8, v4, v20, v0.t
# CHECK-INST: vnclipu.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb8]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclipu.wv v8, v4, v20
# CHECK-INST: vnclipu.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xba]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclipu.wx v8, v4, a0, v0.t
# CHECK-INST: vnclipu.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb8]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclipu.wx v8, v4, a0
# CHECK-INST: vnclipu.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xba]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclipu.wi v8, v4, 31, v0.t
# CHECK-INST: vnclipu.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb8]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclipu.wi v8, v4, 31
# CHECK-INST: vnclipu.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xba]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclip.wv v8, v4, v20, v0.t
# CHECK-INST: vnclip.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xbc]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclip.wv v8, v4, v20
# CHECK-INST: vnclip.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xbe]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclip.wx v8, v4, a0, v0.t
# CHECK-INST: vnclip.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xbc]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclip.wx v8, v4, a0
# CHECK-INST: vnclip.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xbe]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclip.wi v8, v4, 31, v0.t
# CHECK-INST: vnclip.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xbc]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vnclip.wi v8, v4, 31
# CHECK-INST: vnclip.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xbe]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
