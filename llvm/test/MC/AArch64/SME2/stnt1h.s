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


stnt1h  {z0.h, z8.h}, pn8, [x0, x0, lsl #1]  // 10100001-00100000-00100000-00001000
// CHECK-INST: stnt1h  { z0.h, z8.h }, pn8, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x08,0x20,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1202008 <unknown>

stnt1h  {z21.h, z29.h}, pn13, [x10, x21, lsl #1]  // 10100001-00110101-00110101-01011101
// CHECK-INST: stnt1h  { z21.h, z29.h }, pn13, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x5d,0x35,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135355d <unknown>

stnt1h  {z23.h, z31.h}, pn11, [x13, x8, lsl #1]  // 10100001-00101000-00101101-10111111
// CHECK-INST: stnt1h  { z23.h, z31.h }, pn11, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xbf,0x2d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1282dbf <unknown>

stnt1h  {z23.h, z31.h}, pn15, [sp, xzr, lsl #1]  // 10100001-00111111-00111111-11111111
// CHECK-INST: stnt1h  { z23.h, z31.h }, pn15, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xff,0x3f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f3fff <unknown>


stnt1h  {z0.h, z8.h}, pn8, [x0]  // 10100001-01100000-00100000-00001000
// CHECK-INST: stnt1h  { z0.h, z8.h }, pn8, [x0]
// CHECK-ENCODING: [0x08,0x20,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1602008 <unknown>

stnt1h  {z21.h, z29.h}, pn13, [x10, #10, mul vl]  // 10100001-01100101-00110101-01011101
// CHECK-INST: stnt1h  { z21.h, z29.h }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x35,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165355d <unknown>

stnt1h  {z23.h, z31.h}, pn11, [x13, #-16, mul vl]  // 10100001-01101000-00101101-10111111
// CHECK-INST: stnt1h  { z23.h, z31.h }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x2d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1682dbf <unknown>

stnt1h  {z23.h, z31.h}, pn15, [sp, #-2, mul vl]  // 10100001-01101111-00111111-11111111
// CHECK-INST: stnt1h  { z23.h, z31.h }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x3f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f3fff <unknown>


stnt1h  {z0.h, z4.h, z8.h, z12.h}, pn8, [x0, x0, lsl #1]  // 10100001-00100000-10100000-00001000
// CHECK-INST: stnt1h  { z0.h, z4.h, z8.h, z12.h }, pn8, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x08,0xa0,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a120a008 <unknown>

stnt1h  {z17.h, z21.h, z25.h, z29.h}, pn13, [x10, x21, lsl #1]  // 10100001-00110101-10110101-01011001
// CHECK-INST: stnt1h  { z17.h, z21.h, z25.h, z29.h }, pn13, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x59,0xb5,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135b559 <unknown>

stnt1h  {z19.h, z23.h, z27.h, z31.h}, pn11, [x13, x8, lsl #1]  // 10100001-00101000-10101101-10111011
// CHECK-INST: stnt1h  { z19.h, z23.h, z27.h, z31.h }, pn11, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xbb,0xad,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a128adbb <unknown>

stnt1h  {z19.h, z23.h, z27.h, z31.h}, pn15, [sp, xzr, lsl #1]  // 10100001-00111111-10111111-11111011
// CHECK-INST: stnt1h  { z19.h, z23.h, z27.h, z31.h }, pn15, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xfb,0xbf,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13fbffb <unknown>


stnt1h  {z0.h, z4.h, z8.h, z12.h}, pn8, [x0]  // 10100001-01100000-10100000-00001000
// CHECK-INST: stnt1h  { z0.h, z4.h, z8.h, z12.h }, pn8, [x0]
// CHECK-ENCODING: [0x08,0xa0,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a160a008 <unknown>

stnt1h  {z17.h, z21.h, z25.h, z29.h}, pn13, [x10, #20, mul vl]  // 10100001-01100101-10110101-01011001
// CHECK-INST: stnt1h  { z17.h, z21.h, z25.h, z29.h }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0xb5,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165b559 <unknown>

stnt1h  {z19.h, z23.h, z27.h, z31.h}, pn11, [x13, #-32, mul vl]  // 10100001-01101000-10101101-10111011
// CHECK-INST: stnt1h  { z19.h, z23.h, z27.h, z31.h }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0xad,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a168adbb <unknown>

stnt1h  {z19.h, z23.h, z27.h, z31.h}, pn15, [sp, #-4, mul vl]  // 10100001-01101111-10111111-11111011
// CHECK-INST: stnt1h  { z19.h, z23.h, z27.h, z31.h }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0xbf,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16fbffb <unknown>

