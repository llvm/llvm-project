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

usmopa  za0.s, p0/m, p0/m, z0.b, z0.b
// CHECK-INST: usmopa  za0.s, p0/m, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x80,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a1800000 <unknown>

usmopa  za1.s, p5/m, p2/m, z10.b, z21.b
// CHECK-INST: usmopa  za1.s, p5/m, p2/m, z10.b, z21.b
// CHECK-ENCODING: [0x41,0x55,0x95,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a1955541 <unknown>

usmopa  za3.s, p3/m, p7/m, z13.b, z8.b
// CHECK-INST: usmopa  za3.s, p3/m, p7/m, z13.b, z8.b
// CHECK-ENCODING: [0xa3,0xed,0x88,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a188eda3 <unknown>

usmopa  za3.s, p7/m, p7/m, z31.b, z31.b
// CHECK-INST: usmopa  za3.s, p7/m, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xe3,0xff,0x9f,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a19fffe3 <unknown>

usmopa  za1.s, p3/m, p0/m, z17.b, z16.b
// CHECK-INST: usmopa  za1.s, p3/m, p0/m, z17.b, z16.b
// CHECK-ENCODING: [0x21,0x0e,0x90,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a1900e21 <unknown>

usmopa  za1.s, p1/m, p4/m, z1.b, z30.b
// CHECK-INST: usmopa  za1.s, p1/m, p4/m, z1.b, z30.b
// CHECK-ENCODING: [0x21,0x84,0x9e,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a19e8421 <unknown>

usmopa  za0.s, p5/m, p2/m, z19.b, z20.b
// CHECK-INST: usmopa  za0.s, p5/m, p2/m, z19.b, z20.b
// CHECK-ENCODING: [0x60,0x56,0x94,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a1945660 <unknown>

usmopa  za0.s, p6/m, p0/m, z12.b, z2.b
// CHECK-INST: usmopa  za0.s, p6/m, p0/m, z12.b, z2.b
// CHECK-ENCODING: [0x80,0x19,0x82,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a1821980 <unknown>

usmopa  za1.s, p2/m, p6/m, z1.b, z26.b
// CHECK-INST: usmopa  za1.s, p2/m, p6/m, z1.b, z26.b
// CHECK-ENCODING: [0x21,0xc8,0x9a,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a19ac821 <unknown>

usmopa  za1.s, p2/m, p0/m, z22.b, z30.b
// CHECK-INST: usmopa  za1.s, p2/m, p0/m, z22.b, z30.b
// CHECK-ENCODING: [0xc1,0x0a,0x9e,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a19e0ac1 <unknown>

usmopa  za2.s, p5/m, p7/m, z9.b, z1.b
// CHECK-INST: usmopa  za2.s, p5/m, p7/m, z9.b, z1.b
// CHECK-ENCODING: [0x22,0xf5,0x81,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a181f522 <unknown>

usmopa  za3.s, p2/m, p5/m, z12.b, z11.b
// CHECK-INST: usmopa  za3.s, p2/m, p5/m, z12.b, z11.b
// CHECK-ENCODING: [0x83,0xa9,0x8b,0xa1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a18ba983 <unknown>
