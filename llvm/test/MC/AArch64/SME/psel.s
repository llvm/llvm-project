// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// --------------------------------------------------------------------------//
// 8-bit

psel    p0, p0, p0.b[w12, 0]
// CHECK-INST: psel    p0, p0, p0.b[w12, 0]
// CHECK-ENCODING: [0x00,0x40,0x24,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25244000 <unknown>

psel    p5, p5, p10.b[w13, 6]
// CHECK-INST: psel    p5, p5, p10.b[w13, 6]
// CHECK-ENCODING: [0x45,0x55,0x75,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25755545 <unknown>

psel    p7, p11, p13.b[w12, 5]
// CHECK-INST: psel    p7, p11, p13.b[w12, 5]
// CHECK-ENCODING: [0xa7,0x6d,0x6c,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 256c6da7 <unknown>

psel    p15, p15, p15.b[w15, 15]
// CHECK-INST: psel    p15, p15, p15.b[w15, 15]
// CHECK-ENCODING: [0xef,0x7d,0xff,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25ff7def <unknown>

// --------------------------------------------------------------------------//
// 16-bit

psel    p0, p0, p0.h[w12, 0]
// CHECK-INST: psel    p0, p0, p0.h[w12, 0]
// CHECK-ENCODING: [0x00,0x40,0x28,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25284000 <unknown>

psel    p5, p5, p10.h[w13, 3]
// CHECK-INST: psel    p5, p5, p10.h[w13, 3]
// CHECK-ENCODING: [0x45,0x55,0x79,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25795545 <unknown>

psel    p7, p11, p13.h[w12, 2]
// CHECK-INST: psel    p7, p11, p13.h[w12, 2]
// CHECK-ENCODING: [0xa7,0x6d,0x68,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25686da7 <unknown>

psel    p15, p15, p15.h[w15, 7]
// CHECK-INST: psel    p15, p15, p15.h[w15, 7]
// CHECK-ENCODING: [0xef,0x7d,0xfb,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25fb7def <unknown>

// --------------------------------------------------------------------------//
// 32-bit

psel    p0, p0, p0.s[w12, 0]
// CHECK-INST: psel    p0, p0, p0.s[w12, 0]
// CHECK-ENCODING: [0x00,0x40,0x30,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25304000 <unknown>

psel    p5, p5, p10.s[w13, 1]
// CHECK-INST: psel    p5, p5, p10.s[w13, 1]
// CHECK-ENCODING: [0x45,0x55,0x71,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25715545 <unknown>

psel    p7, p11, p13.s[w12, 1]
// CHECK-INST: psel    p7, p11, p13.s[w12, 1]
// CHECK-ENCODING: [0xa7,0x6d,0x70,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25706da7 <unknown>

psel    p15, p15, p15.s[w15, 3]
// CHECK-INST: psel    p15, p15, p15.s[w15, 3]
// CHECK-ENCODING: [0xef,0x7d,0xf3,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25f37def <unknown>

// --------------------------------------------------------------------------//
// 64-bit

psel    p0, p0, p0.d[w12, 0]
// CHECK-INST: psel    p0, p0, p0.d[w12, 0]
// CHECK-ENCODING: [0x00,0x40,0x60,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25604000 <unknown>

psel    p5, p5, p10.d[w13, 0]
// CHECK-INST: psel    p5, p5, p10.d[w13, 0]
// CHECK-ENCODING: [0x45,0x55,0x61,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25615545 <unknown>

psel    p7, p11, p13.d[w12, 0]
// CHECK-INST: psel    p7, p11, p13.d[w12, 0]
// CHECK-ENCODING: [0xa7,0x6d,0x60,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25606da7 <unknown>

psel    p15, p15, p15.d[w15, 1]
// CHECK-INST: psel    p15, p15, p15.d[w15, 1]
// CHECK-ENCODING: [0xef,0x7d,0xe3,0x25]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25e37def <unknown>
