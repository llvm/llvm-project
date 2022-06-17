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

st1b    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-INST: st1b    { z0.s }, p0, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 40 e4 <unknown>

st1b    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-INST: st1b    { z0.s }, p0, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 40 e4 <unknown>

st1b    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-INST: st1b    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x00,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 00 e4 <unknown>

st1b    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-INST: st1b    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x00,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 00 e4 <unknown>

st1b    { z0.d }, p0, [x0, z0.d]
// CHECK-INST: st1b    { z0.d }, p0, [x0, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 00 e4 <unknown>

st1b    { z31.s }, p7, [z31.s, #31]
// CHECK-INST: st1b    { z31.s }, p7, [z31.s, #31]
// CHECK-ENCODING: [0xff,0xbf,0x7f,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 7f e4 <unknown>

st1b    { z31.d }, p7, [z31.d, #31]
// CHECK-INST: st1b    { z31.d }, p7, [z31.d, #31]
// CHECK-ENCODING: [0xff,0xbf,0x5f,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 5f e4 <unknown>

st1b    { z0.s }, p7, [z0.s, #0]
// CHECK-INST: st1b    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0x60,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc 60 e4 <unknown>

st1b    { z0.s }, p7, [z0.s]
// CHECK-INST: st1b    { z0.s }, p7, [z0.s]
// CHECK-ENCODING: [0x00,0xbc,0x60,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc 60 e4 <unknown>

st1b    { z0.d }, p7, [z0.d, #0]
// CHECK-INST: st1b    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc 40 e4 <unknown>

st1b    { z0.d }, p7, [z0.d]
// CHECK-INST: st1b    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc 40 e4 <unknown>
