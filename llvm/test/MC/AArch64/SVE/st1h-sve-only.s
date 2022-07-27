// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

st1h    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x80,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4c08000 <unknown>

st1h    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4c0c000 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4808000 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e480c000 <unknown>

st1h    { z0.s }, p0, [x0, z0.s, uxtw #1]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, uxtw #1]
// CHECK-ENCODING: [0x00,0x80,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4e08000 <unknown>

st1h    { z0.s }, p0, [x0, z0.s, sxtw #1]
// CHECK-INST: st1h    { z0.s }, p0, [x0, z0.s, sxtw #1]
// CHECK-ENCODING: [0x00,0xc0,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4e0c000 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, uxtw #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4a08000 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, sxtw #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4a0c000 <unknown>

st1h    { z0.d }, p0, [x0, z0.d]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e480a000 <unknown>

st1h    { z0.d }, p0, [x0, z0.d, lsl #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, z0.d, lsl #1]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4a0a000 <unknown>

st1h    { z31.s }, p7, [z31.s, #62]
// CHECK-INST: st1h    { z31.s }, p7, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xbf,0xff,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4ffbfff <unknown>

st1h    { z31.d }, p7, [z31.d, #62]
// CHECK-INST: st1h    { z31.d }, p7, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4dfbfff <unknown>

st1h    { z0.s }, p7, [z0.s, #0]
// CHECK-INST: st1h    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4e0bc00 <unknown>

st1h    { z0.s }, p7, [z0.s]
// CHECK-INST: st1h    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4e0bc00 <unknown>

st1h    { z0.d }, p7, [z0.d, #0]
// CHECK-INST: st1h    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4c0bc00 <unknown>

st1h    { z0.d }, p7, [z0.d]
// CHECK-INST: st1h    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e4c0bc00 <unknown>
