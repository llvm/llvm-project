// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

stnt1d  {z0.d-z1.d}, pn8, [x0, x0, lsl #3]  // 10100000-00100000-01100000-00000001
// CHECK-INST: stnt1d  { z0.d, z1.d }, pn8, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x01,0x60,0x20,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0206001 <unknown>

stnt1d  {z20.d-z21.d}, pn13, [x10, x21, lsl #3]  // 10100000-00110101-01110101-01010101
// CHECK-INST: stnt1d  { z20.d, z21.d }, pn13, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x55,0x75,0x35,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0357555 <unknown>

stnt1d  {z22.d-z23.d}, pn11, [x13, x8, lsl #3]  // 10100000-00101000-01101101-10110111
// CHECK-INST: stnt1d  { z22.d, z23.d }, pn11, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb7,0x6d,0x28,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0286db7 <unknown>

stnt1d  {z30.d-z31.d}, pn15, [sp, xzr, lsl #3]  // 10100000-00111111-01111111-11111111
// CHECK-INST: stnt1d  { z30.d, z31.d }, pn15, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a03f7fff <unknown>

stnt1d  {z0.d-z1.d}, pn8, [x0]  // 10100000-01100000-01100000-00000001
// CHECK-INST: stnt1d  { z0.d, z1.d }, pn8, [x0]
// CHECK-ENCODING: [0x01,0x60,0x60,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0606001 <unknown>

stnt1d  {z20.d-z21.d}, pn13, [x10, #10, mul vl]  // 10100000-01100101-01110101-01010101
// CHECK-INST: stnt1d  { z20.d, z21.d }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x75,0x65,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0657555 <unknown>

stnt1d  {z22.d-z23.d}, pn11, [x13, #-16, mul vl]  // 10100000-01101000-01101101-10110111
// CHECK-INST: stnt1d  { z22.d, z23.d }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x6d,0x68,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0686db7 <unknown>

stnt1d  {z30.d-z31.d}, pn15, [sp, #-2, mul vl]  // 10100000-01101111-01111111-11111111
// CHECK-INST: stnt1d  { z30.d, z31.d }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x7f,0x6f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a06f7fff <unknown>

stnt1d  {z0.d-z3.d}, pn8, [x0, x0, lsl #3]  // 10100000-00100000-11100000-00000001
// CHECK-INST: stnt1d  { z0.d - z3.d }, pn8, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x01,0xe0,0x20,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a020e001 <unknown>

stnt1d  {z20.d-z23.d}, pn13, [x10, x21, lsl #3]  // 10100000-00110101-11110101-01010101
// CHECK-INST: stnt1d  { z20.d - z23.d }, pn13, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x55,0xf5,0x35,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a035f555 <unknown>

stnt1d  {z20.d-z23.d}, pn11, [x13, x8, lsl #3]  // 10100000-00101000-11101101-10110101
// CHECK-INST: stnt1d  { z20.d - z23.d }, pn11, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb5,0xed,0x28,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a028edb5 <unknown>

stnt1d  {z28.d-z31.d}, pn15, [sp, xzr, lsl #3]  // 10100000-00111111-11111111-11111101
// CHECK-INST: stnt1d  { z28.d - z31.d }, pn15, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xfd,0xff,0x3f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a03ffffd <unknown>

stnt1d  {z0.d-z3.d}, pn8, [x0]  // 10100000-01100000-11100000-00000001
// CHECK-INST: stnt1d  { z0.d - z3.d }, pn8, [x0]
// CHECK-ENCODING: [0x01,0xe0,0x60,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a060e001 <unknown>

stnt1d  {z20.d-z23.d}, pn13, [x10, #20, mul vl]  // 10100000-01100101-11110101-01010101
// CHECK-INST: stnt1d  { z20.d - z23.d }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x65,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a065f555 <unknown>

stnt1d  {z20.d-z23.d}, pn11, [x13, #-32, mul vl]  // 10100000-01101000-11101101-10110101
// CHECK-INST: stnt1d  { z20.d - z23.d }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb5,0xed,0x68,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a068edb5 <unknown>

stnt1d  {z28.d-z31.d}, pn15, [sp, #-4, mul vl]  // 10100000-01101111-11111111-11111101
// CHECK-INST: stnt1d  { z28.d - z31.d }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfd,0xff,0x6f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a06ffffd <unknown>
