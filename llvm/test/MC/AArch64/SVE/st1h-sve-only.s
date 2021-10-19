// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

st1h    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x80,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 c0 e4 <unknown>

st1h    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 c0 e4 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 80 e4 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 80 e4 <unknown>

st1h    { z0.s }, p0, [x0, z0.s, uxtw #1]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, uxtw #1]
// CHECK-ENCODING: [0x00,0x80,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 e0 e4 <unknown>

st1h    { z0.s }, p0, [x0, z0.s, sxtw #1]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, sxtw #1]
// CHECK-ENCODING: [0x00,0xc0,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 e0 e4 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, uxtw #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a0 e4 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, sxtw #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 a0 e4 <unknown>

st1h    { z0.d }, p0, [x0, z0.d]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 80 e4 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, lsl #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, lsl #1]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a0 e4 <unknown>

st1h    { z31.s }, p7, [z31.s, #62]
// CHECK-INST: st1h    { z31.s }, p7, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xbf,0xff,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf ff e4 <unknown>

st1h    { z31.d }, p7, [z31.d, #62]
// CHECK-INST: st1h    { z31.d }, p7, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf df e4 <unknown>

st1h    { z0.s }, p7, [z0.s, #0]
// CHECK-INST: st1h    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc e0 e4 <unknown>

st1h    { z0.s }, p7, [z0.s]
// CHECK-INST: st1h    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc e0 e4 <unknown>

st1h    { z0.d }, p7, [z0.d, #0]
// CHECK-INST: st1h    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc c0 e4 <unknown>

st1h    { z0.d }, p7, [z0.d]
// CHECK-INST: st1h    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc c0 e4 <unknown>
