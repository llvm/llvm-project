// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// --------------------------------------------------------------------------//
// 8-bit

dup     p0.b, p0/z, p0.b[w12]
// CHECK-INST: dup     p0.b, p0/z, p0.b[w12]
// CHECK-ENCODING: [0x00,0x40,0x24,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 40 24 25 <unknown>

dup     p5.b, p5/z, p10.b[w13, #6]
// CHECK-INST: dup     p5.b, p5/z, p10.b[w13, #6]
// CHECK-ENCODING: [0x45,0x55,0x75,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 75 25 <unknown>

dup     p7.b, p11/z, p13.b[w12, #5]
// CHECK-INST: dup     p7.b, p11/z, p13.b[w12, #5]
// CHECK-ENCODING: [0xa7,0x6d,0x6c,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 6c 25 <unknown>

dup     p15.b, p15/z, p15.b[w15, #15]
// CHECK-INST: dup     p15.b, p15/z, p15.b[w15, #15]
// CHECK-ENCODING: [0xef,0x7d,0xff,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7d ff 25 <unknown>

// --------------------------------------------------------------------------//
// 16-bit

dup     p0.h, p0/z, p0.h[w12]
// CHECK-INST: dup     p0.h, p0/z, p0.h[w12]
// CHECK-ENCODING: [0x00,0x40,0x28,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 40 28 25 <unknown>

dup     p5.h, p5/z, p10.h[w13, #3]
// CHECK-INST: dup     p5.h, p5/z, p10.h[w13, #3]
// CHECK-ENCODING: [0x45,0x55,0x79,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 79 25 <unknown>

dup     p7.h, p11/z, p13.h[w12, #2]
// CHECK-INST: dup     p7.h, p11/z, p13.h[w12, #2]
// CHECK-ENCODING: [0xa7,0x6d,0x68,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 68 25 <unknown>

dup     p15.h, p15/z, p15.h[w15, #7]
// CHECK-INST: dup     p15.h, p15/z, p15.h[w15, #7]
// CHECK-ENCODING: [0xef,0x7d,0xfb,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7d fb 25 <unknown>

// --------------------------------------------------------------------------//
// 32-bit

dup     p0.s, p0/z, p0.s[w12]
// CHECK-INST: dup     p0.s, p0/z, p0.s[w12]
// CHECK-ENCODING: [0x00,0x40,0x30,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 40 30 25 <unknown>

dup     p5.s, p5/z, p10.s[w13, #1]
// CHECK-INST: dup     p5.s, p5/z, p10.s[w13, #1]
// CHECK-ENCODING: [0x45,0x55,0x71,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 71 25 <unknown>

dup     p7.s, p11/z, p13.s[w12, #1]
// CHECK-INST: dup     p7.s, p11/z, p13.s[w12, #1]
// CHECK-ENCODING: [0xa7,0x6d,0x70,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 70 25 <unknown>

dup     p15.s, p15/z, p15.s[w15, #3]
// CHECK-INST: dup     p15.s, p15/z, p15.s[w15, #3]
// CHECK-ENCODING: [0xef,0x7d,0xf3,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7d f3 25 <unknown>

// --------------------------------------------------------------------------//
// 64-bit

dup     p0.d, p0/z, p0.d[w12]
// CHECK-INST: dup     p0.d, p0/z, p0.d[w12]
// CHECK-ENCODING: [0x00,0x40,0x60,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 40 60 25 <unknown>

dup     p5.d, p5/z, p10.d[w13]
// CHECK-INST: dup     p5.d, p5/z, p10.d[w13]
// CHECK-ENCODING: [0x45,0x55,0x61,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 55 61 25 <unknown>

dup     p7.d, p11/z, p13.d[w12]
// CHECK-INST: dup     p7.d, p11/z, p13.d[w12]
// CHECK-ENCODING: [0xa7,0x6d,0x60,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 6d 60 25 <unknown>

dup     p15.d, p15/z, p15.d[w15, #1]
// CHECK-INST: dup     p15.d, p15/z, p15.d[w15, #1]
// CHECK-ENCODING: [0xef,0x7d,0xe3,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 7d e3 25 <unknown>
