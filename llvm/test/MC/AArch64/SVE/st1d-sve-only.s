// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

st1d    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-INST: st1d    { z0.d }, p0, [x0, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0x80,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 80 e5 <unknown>

st1d    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-INST: st1d    { z0.d }, p0, [x0, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xc0,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 80 e5 <unknown>

st1d    { z0.d }, p0, [x0, z0.d, uxtw #3]
// CHECK-INST: st1d    { z0.d }, p0, [x0, z0.d, uxtw #3]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a0 e5 <unknown>

st1d    { z0.d }, p0, [x0, z0.d, sxtw #3]
// CHECK-INST: st1d    { z0.d }, p0, [x0, z0.d, sxtw #3]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 a0 e5 <unknown>

st1d    { z0.d }, p0, [x0, z0.d]
// CHECK-INST: st1d    { z0.d }, p0, [x0, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 80 e5 <unknown>

st1d    { z0.d }, p0, [x0, z0.d, lsl #3]
// CHECK-INST: st1d    { z0.d }, p0, [x0, z0.d, lsl #3]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a0 e5 <unknown>

st1d    { z31.d }, p7, [z31.d, #248]
// CHECK-INST: st1d    { z31.d }, p7, [z31.d, #248]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf df e5 <unknown>

st1d    { z0.d }, p7, [z0.d, #0]
// CHECK-INST: st1d    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc c0 e5 <unknown>

st1d    { z0.d }, p7, [z0.d]
// CHECK-INST: st1d    { z0.d }, p7, [z0.d]
// CHECK-ENCODING: [0x00,0xbc,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 bc c0 e5 <unknown>
