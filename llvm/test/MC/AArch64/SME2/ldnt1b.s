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


ldnt1b  {z0.b, z8.b}, pn8/z, [x0, x0]  // 10100001-00000000-00000000-00001000
// CHECK-INST: ldnt1b  { z0.b, z8.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x08,0x00,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1000008 <unknown>

ldnt1b  {z21.b, z29.b}, pn13/z, [x10, x21]  // 10100001-00010101-00010101-01011101
// CHECK-INST: ldnt1b  { z21.b, z29.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x5d,0x15,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a115155d <unknown>

ldnt1b  {z23.b, z31.b}, pn11/z, [x13, x8]  // 10100001-00001000-00001101-10111111
// CHECK-INST: ldnt1b  { z23.b, z31.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xbf,0x0d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1080dbf <unknown>

ldnt1b  {z23.b, z31.b}, pn15/z, [sp, xzr]  // 10100001-00011111-00011111-11111111
// CHECK-INST: ldnt1b  { z23.b, z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xff,0x1f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f1fff <unknown>


ldnt1b  {z0.b, z8.b}, pn8/z, [x0]  // 10100001-01000000-00000000-00001000
// CHECK-INST: ldnt1b  { z0.b, z8.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x08,0x00,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1400008 <unknown>

ldnt1b  {z21.b, z29.b}, pn13/z, [x10, #10, mul vl]  // 10100001-01000101-00010101-01011101
// CHECK-INST: ldnt1b  { z21.b, z29.b }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x15,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a145155d <unknown>

ldnt1b  {z23.b, z31.b}, pn11/z, [x13, #-16, mul vl]  // 10100001-01001000-00001101-10111111
// CHECK-INST: ldnt1b  { z23.b, z31.b }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x0d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1480dbf <unknown>

ldnt1b  {z23.b, z31.b}, pn15/z, [sp, #-2, mul vl]  // 10100001-01001111-00011111-11111111
// CHECK-INST: ldnt1b  { z23.b, z31.b }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x1f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f1fff <unknown>


ldnt1b  {z0.b, z4.b, z8.b, z12.b}, pn8/z, [x0, x0]  // 10100001-00000000-10000000-00001000
// CHECK-INST: ldnt1b  { z0.b, z4.b, z8.b, z12.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x08,0x80,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1008008 <unknown>

ldnt1b  {z17.b, z21.b, z25.b, z29.b}, pn13/z, [x10, x21]  // 10100001-00010101-10010101-01011001
// CHECK-INST: ldnt1b  { z17.b, z21.b, z25.b, z29.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x59,0x95,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1159559 <unknown>

ldnt1b  {z19.b, z23.b, z27.b, z31.b}, pn11/z, [x13, x8]  // 10100001-00001000-10001101-10111011
// CHECK-INST: ldnt1b  { z19.b, z23.b, z27.b, z31.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xbb,0x8d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1088dbb <unknown>

ldnt1b  {z19.b, z23.b, z27.b, z31.b}, pn15/z, [sp, xzr]  // 10100001-00011111-10011111-11111011
// CHECK-INST: ldnt1b  { z19.b, z23.b, z27.b, z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xfb,0x9f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f9ffb <unknown>


ldnt1b  {z0.b, z4.b, z8.b, z12.b}, pn8/z, [x0]  // 10100001-01000000-10000000-00001000
// CHECK-INST: ldnt1b  { z0.b, z4.b, z8.b, z12.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x08,0x80,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1408008 <unknown>

ldnt1b  {z17.b, z21.b, z25.b, z29.b}, pn13/z, [x10, #20, mul vl]  // 10100001-01000101-10010101-01011001
// CHECK-INST: ldnt1b  { z17.b, z21.b, z25.b, z29.b }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0x95,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1459559 <unknown>

ldnt1b  {z19.b, z23.b, z27.b, z31.b}, pn11/z, [x13, #-32, mul vl]  // 10100001-01001000-10001101-10111011
// CHECK-INST: ldnt1b  { z19.b, z23.b, z27.b, z31.b }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0x8d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1488dbb <unknown>

ldnt1b  {z19.b, z23.b, z27.b, z31.b}, pn15/z, [sp, #-4, mul vl]  // 10100001-01001111-10011111-11111011
// CHECK-INST: ldnt1b  { z19.b, z23.b, z27.b, z31.b }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0x9f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f9ffb <unknown>

