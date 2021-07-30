// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

insr    z0.b, w0
// CHECK-INST: insr    z0.b, w0
// CHECK-ENCODING: [0x00,0x38,0x24,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 38 24 05 <unknown>

insr    z0.h, w0
// CHECK-INST: insr    z0.h, w0
// CHECK-ENCODING: [0x00,0x38,0x64,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 38 64 05 <unknown>

insr    z0.s, w0
// CHECK-INST: insr    z0.s, w0
// CHECK-ENCODING: [0x00,0x38,0xa4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 38 a4 05 <unknown>

insr    z0.d, x0
// CHECK-INST: insr    z0.d, x0
// CHECK-ENCODING: [0x00,0x38,0xe4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 38 e4 05 <unknown>

insr    z31.b, wzr
// CHECK-INST: insr    z31.b, wzr
// CHECK-ENCODING: [0xff,0x3b,0x24,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b 24 05 <unknown>

insr    z31.h, wzr
// CHECK-INST: insr    z31.h, wzr
// CHECK-ENCODING: [0xff,0x3b,0x64,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b 64 05 <unknown>

insr    z31.s, wzr
// CHECK-INST: insr    z31.s, wzr
// CHECK-ENCODING: [0xff,0x3b,0xa4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b a4 05 <unknown>

insr    z31.d, xzr
// CHECK-INST: insr    z31.d, xzr
// CHECK-ENCODING: [0xff,0x3b,0xe4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b e4 05 <unknown>

insr    z31.b, b31
// CHECK-INST: insr    z31.b, b31
// CHECK-ENCODING: [0xff,0x3b,0x34,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b 34 05 <unknown>

insr    z31.h, h31
// CHECK-INST: insr    z31.h, h31
// CHECK-ENCODING: [0xff,0x3b,0x74,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b 74 05 <unknown>

insr    z31.s, s31
// CHECK-INST: insr    z31.s, s31
// CHECK-ENCODING: [0xff,0x3b,0xb4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b b4 05 <unknown>

insr    z31.d, d31
// CHECK-INST: insr    z31.d, d31
// CHECK-ENCODING: [0xff,0x3b,0xf4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b f4 05 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

insr    z31.d, xzr
// CHECK-INST: insr	z31.d, xzr
// CHECK-ENCODING: [0xff,0x3b,0xe4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b e4 05 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

insr    z4.d, d31
// CHECK-INST: insr	z4.d, d31
// CHECK-ENCODING: [0xe4,0x3b,0xf4,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e4 3b f4 05 <unknown>
