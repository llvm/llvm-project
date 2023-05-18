// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

st1w    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-INST: st1w    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5408000 <unknown>

st1w    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-INST: st1w    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e540c000 <unknown>

st1w    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-INST: st1w    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x00,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5008000 <unknown>

st1w    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-INST: st1w    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x00,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e500c000 <unknown>

st1w    { z0.s }, p0, [x0, z0.s, uxtw #2]
// CHECK-INST: st1w    { z0.s }, p0, [x0, z0.s, uxtw #2]
// CHECK-ENCODING: [0x00,0x80,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5608000 <unknown>

st1w    { z0.s }, p0, [x0, z0.s, sxtw #2]
// CHECK-INST: st1w    { z0.s }, p0, [x0, z0.s, sxtw #2]
// CHECK-ENCODING: [0x00,0xc0,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e560c000 <unknown>

st1w    { z0.d }, p0, [x0, z0.d, uxtw #2]
// CHECK-INST: st1w    { z0.d }, p0, [x0, z0.d, uxtw #2]
// CHECK-ENCODING: [0x00,0x80,0x20,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5208000 <unknown>

st1w    { z0.d }, p0, [x0, z0.d, sxtw #2]
// CHECK-INST: st1w    { z0.d }, p0, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0xc0,0x20,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e520c000 <unknown>

st1w    { z0.d }, p0, [x0, z0.d]
// CHECK-INST: st1w    { z0.d }, p0, [x0, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e500a000 <unknown>

st1w    { z0.d }, p0, [x0, z0.d, lsl #2]
// CHECK-INST: st1w    { z0.d }, p0, [x0, z0.d, lsl #2]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e520a000 <unknown>

st1w    { z31.s }, p7, [z31.s, #124]
// CHECK-INST: st1w    { z31.s }, p7, [z31.s, #124]
// CHECK-ENCODING: [0xff,0xbf,0x7f,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e57fbfff <unknown>

st1w    { z31.d }, p7, [z31.d, #124]
// CHECK-INST: st1w    { z31.d }, p7, [z31.d, #124]
// CHECK-ENCODING: [0xff,0xbf,0x5f,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e55fbfff <unknown>

st1w    { z0.s }, p7, [z0.s, #0]
// CHECK-INST: st1w    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e560bc00 <unknown>

st1w    { z0.s }, p7, [z0.s]
// CHECK-INST: st1w    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e560bc00 <unknown>

st1w    { z0.d }, p7, [z0.d, #0]
// CHECK-INST: st1w    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e540bc00 <unknown>

st1w    { z0.d }, p7, [z0.d]
// CHECK-INST: st1w    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e540bc00 <unknown>
