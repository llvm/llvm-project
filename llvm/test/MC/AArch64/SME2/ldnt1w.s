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


ldnt1w  {z0.s, z8.s}, pn8/z, [x0, x0, lsl #2]  // 10100001-00000000-01000000-00001000
// CHECK-INST: ldnt1w  { z0.s, z8.s }, pn8/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x08,0x40,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1004008 <unknown>

ldnt1w  {z21.s, z29.s}, pn13/z, [x10, x21, lsl #2]  // 10100001-00010101-01010101-01011101
// CHECK-INST: ldnt1w  { z21.s, z29.s }, pn13/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x5d,0x55,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a115555d <unknown>

ldnt1w  {z23.s, z31.s}, pn11/z, [x13, x8, lsl #2]  // 10100001-00001000-01001101-10111111
// CHECK-INST: ldnt1w  { z23.s, z31.s }, pn11/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xbf,0x4d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1084dbf <unknown>

ldnt1w  {z23.s, z31.s}, pn15/z, [sp, xzr, lsl #2]  // 10100001-00011111-01011111-11111111
// CHECK-INST: ldnt1w  { z23.s, z31.s }, pn15/z, [sp, xzr, lsl #2]
// CHECK-ENCODING: [0xff,0x5f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f5fff <unknown>


ldnt1w  {z0.s, z8.s}, pn8/z, [x0]  // 10100001-01000000-01000000-00001000
// CHECK-INST: ldnt1w  { z0.s, z8.s }, pn8/z, [x0]
// CHECK-ENCODING: [0x08,0x40,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1404008 <unknown>

ldnt1w  {z21.s, z29.s}, pn13/z, [x10, #10, mul vl]  // 10100001-01000101-01010101-01011101
// CHECK-INST: ldnt1w  { z21.s, z29.s }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x55,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a145555d <unknown>

ldnt1w  {z23.s, z31.s}, pn11/z, [x13, #-16, mul vl]  // 10100001-01001000-01001101-10111111
// CHECK-INST: ldnt1w  { z23.s, z31.s }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x4d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1484dbf <unknown>

ldnt1w  {z23.s, z31.s}, pn15/z, [sp, #-2, mul vl]  // 10100001-01001111-01011111-11111111
// CHECK-INST: ldnt1w  { z23.s, z31.s }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x5f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f5fff <unknown>


ldnt1w  {z0.s, z4.s, z8.s, z12.s}, pn8/z, [x0, x0, lsl #2]  // 10100001-00000000-11000000-00001000
// CHECK-INST: ldnt1w  { z0.s, z4.s, z8.s, z12.s }, pn8/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x08,0xc0,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a100c008 <unknown>

ldnt1w  {z17.s, z21.s, z25.s, z29.s}, pn13/z, [x10, x21, lsl #2]  // 10100001-00010101-11010101-01011001
// CHECK-INST: ldnt1w  { z17.s, z21.s, z25.s, z29.s }, pn13/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x59,0xd5,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a115d559 <unknown>

ldnt1w  {z19.s, z23.s, z27.s, z31.s}, pn11/z, [x13, x8, lsl #2]  // 10100001-00001000-11001101-10111011
// CHECK-INST: ldnt1w  { z19.s, z23.s, z27.s, z31.s }, pn11/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xbb,0xcd,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a108cdbb <unknown>

ldnt1w  {z19.s, z23.s, z27.s, z31.s}, pn15/z, [sp, xzr, lsl #2]  // 10100001-00011111-11011111-11111011
// CHECK-INST: ldnt1w  { z19.s, z23.s, z27.s, z31.s }, pn15/z, [sp, xzr, lsl #2]
// CHECK-ENCODING: [0xfb,0xdf,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11fdffb <unknown>


ldnt1w  {z0.s, z4.s, z8.s, z12.s}, pn8/z, [x0]  // 10100001-01000000-11000000-00001000
// CHECK-INST: ldnt1w  { z0.s, z4.s, z8.s, z12.s }, pn8/z, [x0]
// CHECK-ENCODING: [0x08,0xc0,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a140c008 <unknown>

ldnt1w  {z17.s, z21.s, z25.s, z29.s}, pn13/z, [x10, #20, mul vl]  // 10100001-01000101-11010101-01011001
// CHECK-INST: ldnt1w  { z17.s, z21.s, z25.s, z29.s }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0xd5,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a145d559 <unknown>

ldnt1w  {z19.s, z23.s, z27.s, z31.s}, pn11/z, [x13, #-32, mul vl]  // 10100001-01001000-11001101-10111011
// CHECK-INST: ldnt1w  { z19.s, z23.s, z27.s, z31.s }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0xcd,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a148cdbb <unknown>

ldnt1w  {z19.s, z23.s, z27.s, z31.s}, pn15/z, [sp, #-4, mul vl]  // 10100001-01001111-11011111-11111011
// CHECK-INST: ldnt1w  { z19.s, z23.s, z27.s, z31.s }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0xdf,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14fdffb <unknown>

