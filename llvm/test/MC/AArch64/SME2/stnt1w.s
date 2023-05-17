// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


stnt1w  {z0.s, z8.s}, pn8, [x0, x0, lsl #2]  // 10100001-00100000-01000000-00001000
// CHECK-INST: stnt1w  { z0.s, z8.s }, pn8, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x08,0x40,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1204008 <unknown>

stnt1w  {z21.s, z29.s}, pn13, [x10, x21, lsl #2]  // 10100001-00110101-01010101-01011101
// CHECK-INST: stnt1w  { z21.s, z29.s }, pn13, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x5d,0x55,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135555d <unknown>

stnt1w  {z23.s, z31.s}, pn11, [x13, x8, lsl #2]  // 10100001-00101000-01001101-10111111
// CHECK-INST: stnt1w  { z23.s, z31.s }, pn11, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xbf,0x4d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1284dbf <unknown>

stnt1w  {z23.s, z31.s}, pn15, [sp, xzr, lsl #2]  // 10100001-00111111-01011111-11111111
// CHECK-INST: stnt1w  { z23.s, z31.s }, pn15, [sp, xzr, lsl #2]
// CHECK-ENCODING: [0xff,0x5f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f5fff <unknown>


stnt1w  {z0.s, z8.s}, pn8, [x0]  // 10100001-01100000-01000000-00001000
// CHECK-INST: stnt1w  { z0.s, z8.s }, pn8, [x0]
// CHECK-ENCODING: [0x08,0x40,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1604008 <unknown>

stnt1w  {z21.s, z29.s}, pn13, [x10, #10, mul vl]  // 10100001-01100101-01010101-01011101
// CHECK-INST: stnt1w  { z21.s, z29.s }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x55,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165555d <unknown>

stnt1w  {z23.s, z31.s}, pn11, [x13, #-16, mul vl]  // 10100001-01101000-01001101-10111111
// CHECK-INST: stnt1w  { z23.s, z31.s }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x4d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1684dbf <unknown>

stnt1w  {z23.s, z31.s}, pn15, [sp, #-2, mul vl]  // 10100001-01101111-01011111-11111111
// CHECK-INST: stnt1w  { z23.s, z31.s }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x5f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f5fff <unknown>


stnt1w  {z0.s, z4.s, z8.s, z12.s}, pn8, [x0, x0, lsl #2]  // 10100001-00100000-11000000-00001000
// CHECK-INST: stnt1w  { z0.s, z4.s, z8.s, z12.s }, pn8, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x08,0xc0,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a120c008 <unknown>

stnt1w  {z17.s, z21.s, z25.s, z29.s}, pn13, [x10, x21, lsl #2]  // 10100001-00110101-11010101-01011001
// CHECK-INST: stnt1w  { z17.s, z21.s, z25.s, z29.s }, pn13, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x59,0xd5,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135d559 <unknown>

stnt1w  {z19.s, z23.s, z27.s, z31.s}, pn11, [x13, x8, lsl #2]  // 10100001-00101000-11001101-10111011
// CHECK-INST: stnt1w  { z19.s, z23.s, z27.s, z31.s }, pn11, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xbb,0xcd,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a128cdbb <unknown>

stnt1w  {z19.s, z23.s, z27.s, z31.s}, pn15, [sp, xzr, lsl #2]  // 10100001-00111111-11011111-11111011
// CHECK-INST: stnt1w  { z19.s, z23.s, z27.s, z31.s }, pn15, [sp, xzr, lsl #2]
// CHECK-ENCODING: [0xfb,0xdf,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13fdffb <unknown>


stnt1w  {z0.s, z4.s, z8.s, z12.s}, pn8, [x0]  // 10100001-01100000-11000000-00001000
// CHECK-INST: stnt1w  { z0.s, z4.s, z8.s, z12.s }, pn8, [x0]
// CHECK-ENCODING: [0x08,0xc0,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a160c008 <unknown>

stnt1w  {z17.s, z21.s, z25.s, z29.s}, pn13, [x10, #20, mul vl]  // 10100001-01100101-11010101-01011001
// CHECK-INST: stnt1w  { z17.s, z21.s, z25.s, z29.s }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0xd5,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165d559 <unknown>

stnt1w  {z19.s, z23.s, z27.s, z31.s}, pn11, [x13, #-32, mul vl]  // 10100001-01101000-11001101-10111011
// CHECK-INST: stnt1w  { z19.s, z23.s, z27.s, z31.s }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0xcd,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a168cdbb <unknown>

stnt1w  {z19.s, z23.s, z27.s, z31.s}, pn15, [sp, #-4, mul vl]  // 10100001-01101111-11011111-11111011
// CHECK-INST: stnt1w  { z19.s, z23.s, z27.s, z31.s }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0xdf,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16fdffb <unknown>

