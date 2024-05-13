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


stnt1d  {z0.d, z8.d}, pn8, [x0, x0, lsl #3]  // 10100001-00100000-01100000-00001000
// CHECK-INST: stnt1d  { z0.d, z8.d }, pn8, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x08,0x60,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1206008 <unknown>

stnt1d  {z21.d, z29.d}, pn13, [x10, x21, lsl #3]  // 10100001-00110101-01110101-01011101
// CHECK-INST: stnt1d  { z21.d, z29.d }, pn13, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x5d,0x75,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135755d <unknown>

stnt1d  {z23.d, z31.d}, pn11, [x13, x8, lsl #3]  // 10100001-00101000-01101101-10111111
// CHECK-INST: stnt1d  { z23.d, z31.d }, pn11, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xbf,0x6d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1286dbf <unknown>

stnt1d  {z23.d, z31.d}, pn15, [sp, xzr, lsl #3]  // 10100001-00111111-01111111-11111111
// CHECK-INST: stnt1d  { z23.d, z31.d }, pn15, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f7fff <unknown>


stnt1d  {z0.d, z8.d}, pn8, [x0]  // 10100001-01100000-01100000-00001000
// CHECK-INST: stnt1d  { z0.d, z8.d }, pn8, [x0]
// CHECK-ENCODING: [0x08,0x60,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1606008 <unknown>

stnt1d  {z21.d, z29.d}, pn13, [x10, #10, mul vl]  // 10100001-01100101-01110101-01011101
// CHECK-INST: stnt1d  { z21.d, z29.d }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x5d,0x75,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165755d <unknown>

stnt1d  {z23.d, z31.d}, pn11, [x13, #-16, mul vl]  // 10100001-01101000-01101101-10111111
// CHECK-INST: stnt1d  { z23.d, z31.d }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xbf,0x6d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1686dbf <unknown>

stnt1d  {z23.d, z31.d}, pn15, [sp, #-2, mul vl]  // 10100001-01101111-01111111-11111111
// CHECK-INST: stnt1d  { z23.d, z31.d }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x7f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f7fff <unknown>


stnt1d  {z0.d, z4.d, z8.d, z12.d}, pn8, [x0, x0, lsl #3]  // 10100001-00100000-11100000-00001000
// CHECK-INST: stnt1d  { z0.d, z4.d, z8.d, z12.d }, pn8, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x08,0xe0,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a120e008 <unknown>

stnt1d  {z17.d, z21.d, z25.d, z29.d}, pn13, [x10, x21, lsl #3]  // 10100001-00110101-11110101-01011001
// CHECK-INST: stnt1d  { z17.d, z21.d, z25.d, z29.d }, pn13, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x59,0xf5,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135f559 <unknown>

stnt1d  {z19.d, z23.d, z27.d, z31.d}, pn11, [x13, x8, lsl #3]  // 10100001-00101000-11101101-10111011
// CHECK-INST: stnt1d  { z19.d, z23.d, z27.d, z31.d }, pn11, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xbb,0xed,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a128edbb <unknown>

stnt1d  {z19.d, z23.d, z27.d, z31.d}, pn15, [sp, xzr, lsl #3]  // 10100001-00111111-11111111-11111011
// CHECK-INST: stnt1d  { z19.d, z23.d, z27.d, z31.d }, pn15, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xfb,0xff,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13ffffb <unknown>


stnt1d  {z0.d, z4.d, z8.d, z12.d}, pn8, [x0]  // 10100001-01100000-11100000-00001000
// CHECK-INST: stnt1d  { z0.d, z4.d, z8.d, z12.d }, pn8, [x0]
// CHECK-ENCODING: [0x08,0xe0,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a160e008 <unknown>

stnt1d  {z17.d, z21.d, z25.d, z29.d}, pn13, [x10, #20, mul vl]  // 10100001-01100101-11110101-01011001
// CHECK-INST: stnt1d  { z17.d, z21.d, z25.d, z29.d }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x59,0xf5,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165f559 <unknown>

stnt1d  {z19.d, z23.d, z27.d, z31.d}, pn11, [x13, #-32, mul vl]  // 10100001-01101000-11101101-10111011
// CHECK-INST: stnt1d  { z19.d, z23.d, z27.d, z31.d }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xbb,0xed,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a168edbb <unknown>

stnt1d  {z19.d, z23.d, z27.d, z31.d}, pn15, [sp, #-4, mul vl]  // 10100001-01101111-11111111-11111011
// CHECK-INST: stnt1d  { z19.d, z23.d, z27.d, z31.d }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfb,0xff,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16ffffb <unknown>

