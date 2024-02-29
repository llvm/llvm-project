// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sve2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


st1d    {z0.q}, p0, [x0, x0, lsl #3]  // 11100101-11000000-01000000-00000000
// CHECK-INST: st1d    { z0.q }, p0, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x40,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c04000 <unknown>

st1d    {z21.q}, p5, [x10, x21, lsl #3]  // 11100101-11010101-01010101-01010101
// CHECK-INST: st1d    { z21.q }, p5, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5d55555 <unknown>

st1d    {z23.q}, p3, [x13, x8, lsl #3]  // 11100101-11001000-01001101-10110111
// CHECK-INST: st1d    { z23.q }, p3, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb7,0x4d,0xc8,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c84db7 <unknown>

st1d    z23.q, p3, [x13, x8, lsl #3]  // 11100101-11001000-01001101-10110111
// CHECK-INST: st1d    { z23.q }, p3, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb7,0x4d,0xc8,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c84db7 <unknown>

st1d    {z0.q}, p0, [x0]  // 11100101-11000000-11100000-00000000
// CHECK-INST: st1d    { z0.q }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c0e000 <unknown>

st1d    z0.q, p0, [x0]  // 11100101-11000000-11100000-00000000
// CHECK-INST: st1d    { z0.q }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c0e000 <unknown>

st1d    {z21.q}, p5, [x10, #5, mul vl]  // 11100101-11000101-11110101-01010101
// CHECK-INST: st1d    { z21.q }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xc5,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c5f555 <unknown>

st1d    {z23.q}, p3, [x13, #-8, mul vl]  // 11100101-11001000-11101101-10110111
// CHECK-INST: st1d    { z23.q }, p3, [x13, #-8, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xc8,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5c8edb7 <unknown>

st1d    {z31.q}, p7, [sp, #-1, mul vl]  // 11100101-11001111-11111111-11111111
// CHECK-INST: st1d    { z31.q }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0xcf,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5cfffff <unknown>

st1d    z31.q, p7, [sp, #-1, mul vl]  // 11100101-11001111-11111111-11111111
// CHECK-INST: st1d    { z31.q }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0xcf,0xe5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: e5cfffff <unknown>

