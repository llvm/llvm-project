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


ldnt1h  {z0.h, z8.h}, pn8/z, [x0, x0, lsl #1]  // 10100001-00000000-00100000-00001000
// CHECK-INST: ldnt1h  { z0.h, z8.h }, pn8/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x08,0x20,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1002008 <unknown>

ldnt1h  {z21.h, z29.h}, pn13/z, [x10, x21, lsl #1]  // 10100001-00010101-00110101-01011101
// CHECK-INST: ldnt1h  { z21.h, z29.h }, pn13/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x5d,0x35,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a115355d <unknown>

ldnt1h  {z23.h, z31.h}, pn11/z, [x13, x8, lsl #1]  // 10100001-00001000-00101101-10111111
// CHECK-INST: ldnt1h  { z23.h, z31.h }, pn11/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xbf,0x2d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1082dbf <unknown>

ldnt1h  {z23.h, z31.h}, pn15/z, [sp, xzr, lsl #1]  // 10100001-00011111-00111111-11111111
// CHECK-INST: ldnt1h  { z23.h, z31.h }, pn15/z, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xff,0x3f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f3fff <unknown>


ldnt1h  {z0.h, z8.h}, pn8/z, [x0]  // 10100001-01000000-00100000-00001000
// CHECK-INST: ldnt1h  { z0.h, z8.h }, pn8/z, [x0]
// CHECK-ENCODING: [0x08,0x20,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1402008 <unknown>

ldnt1h  {z21.h, z29.h}, pn13/z, [x10, #10, mul vl]  // 10100001-01000101-00110101-01011101
// CHECK-INST: ldnt1h  { z21.h, z29.h }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x35,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a145355d <unknown>

ldnt1h  {z23.h, z31.h}, pn11/z, [x13, #-16, mul vl]  // 10100001-01001000-00101101-10111111
// CHECK-INST: ldnt1h  { z23.h, z31.h }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x2d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1482dbf <unknown>

ldnt1h  {z23.h, z31.h}, pn15/z, [sp, #-2, mul vl]  // 10100001-01001111-00111111-11111111
// CHECK-INST: ldnt1h  { z23.h, z31.h }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x3f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f3fff <unknown>


ldnt1h  {z0.h, z4.h, z8.h, z12.h}, pn8/z, [x0, x0, lsl #1]  // 10100001-00000000-10100000-00001000
// CHECK-INST: ldnt1h  { z0.h, z4.h, z8.h, z12.h }, pn8/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x08,0xa0,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a100a008 <unknown>

ldnt1h  {z17.h, z21.h, z25.h, z29.h}, pn13/z, [x10, x21, lsl #1]  // 10100001-00010101-10110101-01011001
// CHECK-INST: ldnt1h  { z17.h, z21.h, z25.h, z29.h }, pn13/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x59,0xb5,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a115b559 <unknown>

ldnt1h  {z19.h, z23.h, z27.h, z31.h}, pn11/z, [x13, x8, lsl #1]  // 10100001-00001000-10101101-10111011
// CHECK-INST: ldnt1h  { z19.h, z23.h, z27.h, z31.h }, pn11/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xbb,0xad,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a108adbb <unknown>

ldnt1h  {z19.h, z23.h, z27.h, z31.h}, pn15/z, [sp, xzr, lsl #1]  // 10100001-00011111-10111111-11111011
// CHECK-INST: ldnt1h  { z19.h, z23.h, z27.h, z31.h }, pn15/z, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xfb,0xbf,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11fbffb <unknown>


ldnt1h  {z0.h, z4.h, z8.h, z12.h}, pn8/z, [x0]  // 10100001-01000000-10100000-00001000
// CHECK-INST: ldnt1h  { z0.h, z4.h, z8.h, z12.h }, pn8/z, [x0]
// CHECK-ENCODING: [0x08,0xa0,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a140a008 <unknown>

ldnt1h  {z17.h, z21.h, z25.h, z29.h}, pn13/z, [x10, #20, mul vl]  // 10100001-01000101-10110101-01011001
// CHECK-INST: ldnt1h  { z17.h, z21.h, z25.h, z29.h }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0xb5,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a145b559 <unknown>

ldnt1h  {z19.h, z23.h, z27.h, z31.h}, pn11/z, [x13, #-32, mul vl]  // 10100001-01001000-10101101-10111011
// CHECK-INST: ldnt1h  { z19.h, z23.h, z27.h, z31.h }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0xad,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a148adbb <unknown>

ldnt1h  {z19.h, z23.h, z27.h, z31.h}, pn15/z, [sp, #-4, mul vl]  // 10100001-01001111-10111111-11111011
// CHECK-INST: ldnt1h  { z19.h, z23.h, z27.h, z31.h }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0xbf,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14fbffb <unknown>

