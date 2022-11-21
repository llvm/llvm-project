// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sve2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p1 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


ld1w    {z0.q}, p0/z, [x0, x0, lsl #2]  // 10100101-00000000-10000000-00000000
// CHECK-INST: ld1w    { z0.q }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x80,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5008000 <unknown>

ld1w    {z21.q}, p5/z, [x10, x21, lsl #2]  // 10100101-00010101-10010101-01010101
// CHECK-INST: ld1w    { z21.q }, p5/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x55,0x95,0x15,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5159555 <unknown>

ld1w    {z23.q}, p3/z, [x13, x8, lsl #2]  // 10100101-00001000-10001101-10110111
// CHECK-INST: ld1w    { z23.q }, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xb7,0x8d,0x08,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5088db7 <unknown>

ld1w    z23.q, p3/z, [x13, x8, lsl #2]  // 10100101-00001000-10001101-10110111
// CHECK-INST: ld1w    { z23.q }, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xb7,0x8d,0x08,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5088db7 <unknown>

ld1w    {z0.q}, p0/z, [x0]  // 10100101-00010000-00100000-00000000
// CHECK-INST: ld1w    { z0.q }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x20,0x10,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5102000 <unknown>

ld1w    {z21.q}, p5/z, [x10, #5, mul vl]  // 10100101-00010101-00110101-01010101
// CHECK-INST: ld1w    { z21.q }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0x35,0x15,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5153555 <unknown>

ld1w    {z23.q}, p3/z, [x13, #-8, mul vl]  // 10100101-00011000-00101101-10110111
// CHECK-INST: ld1w    { z23.q }, p3/z, [x13, #-8, mul vl]
// CHECK-ENCODING: [0xb7,0x2d,0x18,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a5182db7 <unknown>

ld1w    {z31.q}, p7/z, [sp, #-1, mul vl]  // 10100101-00011111-00111111-11111111
// CHECK-INST: ld1w    { z31.q }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0x3f,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sve2p1
// CHECK-UNKNOWN: a51f3fff <unknown>

