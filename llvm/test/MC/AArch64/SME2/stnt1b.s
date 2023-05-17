// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


stnt1b  {z0.b, z8.b}, pn8, [x0, x0]  // 10100001-00100000-00000000-00001000
// CHECK-INST: stnt1b  { z0.b, z8.b }, pn8, [x0, x0]
// CHECK-ENCODING: [0x08,0x00,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1200008 <unknown>

stnt1b  {z21.b, z29.b}, pn13, [x10, x21]  // 10100001-00110101-00010101-01011101
// CHECK-INST: stnt1b  { z21.b, z29.b }, pn13, [x10, x21]
// CHECK-ENCODING: [0x5d,0x15,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135155d <unknown>

stnt1b  {z23.b, z31.b}, pn11, [x13, x8]  // 10100001-00101000-00001101-10111111
// CHECK-INST: stnt1b  { z23.b, z31.b }, pn11, [x13, x8]
// CHECK-ENCODING: [0xbf,0x0d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1280dbf <unknown>

stnt1b  {z23.b, z31.b}, pn15, [sp, xzr]  // 10100001-00111111-00011111-11111111
// CHECK-INST: stnt1b  { z23.b, z31.b }, pn15, [sp, xzr]
// CHECK-ENCODING: [0xff,0x1f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f1fff <unknown>


stnt1b  {z0.b, z8.b}, pn8, [x0]  // 10100001-01100000-00000000-00001000
// CHECK-INST: stnt1b  { z0.b, z8.b }, pn8, [x0]
// CHECK-ENCODING: [0x08,0x00,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1600008 <unknown>

stnt1b  {z21.b, z29.b}, pn13, [x10, #10, mul vl]  // 10100001-01100101-00010101-01011101
// CHECK-INST: stnt1b  { z21.b, z29.b }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x15,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165155d <unknown>

stnt1b  {z23.b, z31.b}, pn11, [x13, #-16, mul vl]  // 10100001-01101000-00001101-10111111
// CHECK-INST: stnt1b  { z23.b, z31.b }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x0d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1680dbf <unknown>

stnt1b  {z23.b, z31.b}, pn15, [sp, #-2, mul vl]  // 10100001-01101111-00011111-11111111
// CHECK-INST: stnt1b  { z23.b, z31.b }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x1f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f1fff <unknown>


stnt1b  {z0.b, z4.b, z8.b, z12.b}, pn8, [x0, x0]  // 10100001-00100000-10000000-00001000
// CHECK-INST: stnt1b  { z0.b, z4.b, z8.b, z12.b }, pn8, [x0, x0]
// CHECK-ENCODING: [0x08,0x80,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1208008 <unknown>

stnt1b  {z17.b, z21.b, z25.b, z29.b}, pn13, [x10, x21]  // 10100001-00110101-10010101-01011001
// CHECK-INST: stnt1b  { z17.b, z21.b, z25.b, z29.b }, pn13, [x10, x21]
// CHECK-ENCODING: [0x59,0x95,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1359559 <unknown>

stnt1b  {z19.b, z23.b, z27.b, z31.b}, pn11, [x13, x8]  // 10100001-00101000-10001101-10111011
// CHECK-INST: stnt1b  { z19.b, z23.b, z27.b, z31.b }, pn11, [x13, x8]
// CHECK-ENCODING: [0xbb,0x8d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1288dbb <unknown>

stnt1b  {z19.b, z23.b, z27.b, z31.b}, pn15, [sp, xzr]  // 10100001-00111111-10011111-11111011
// CHECK-INST: stnt1b  { z19.b, z23.b, z27.b, z31.b }, pn15, [sp, xzr]
// CHECK-ENCODING: [0xfb,0x9f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f9ffb <unknown>


stnt1b  {z0.b, z4.b, z8.b, z12.b}, pn8, [x0]  // 10100001-01100000-10000000-00001000
// CHECK-INST: stnt1b  { z0.b, z4.b, z8.b, z12.b }, pn8, [x0]
// CHECK-ENCODING: [0x08,0x80,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1608008 <unknown>

stnt1b  {z17.b, z21.b, z25.b, z29.b}, pn13, [x10, #20, mul vl]  // 10100001-01100101-10010101-01011001
// CHECK-INST: stnt1b  { z17.b, z21.b, z25.b, z29.b }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0x95,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1659559 <unknown>

stnt1b  {z19.b, z23.b, z27.b, z31.b}, pn11, [x13, #-32, mul vl]  // 10100001-01101000-10001101-10111011
// CHECK-INST: stnt1b  { z19.b, z23.b, z27.b, z31.b }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0x8d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1688dbb <unknown>

stnt1b  {z19.b, z23.b, z27.b, z31.b}, pn15, [sp, #-4, mul vl]  // 10100001-01101111-10011111-11111011
// CHECK-INST: stnt1b  { z19.b, z23.b, z27.b, z31.b }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0x9f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f9ffb <unknown>

