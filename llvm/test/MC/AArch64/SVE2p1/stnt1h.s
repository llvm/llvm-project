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

stnt1h  {z0.h-z1.h}, pn8, [x0, x0, lsl #1]  // 10100000-00100000-00100000-00000001
// CHECK-INST: stnt1h  { z0.h, z1.h }, pn8, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x01,0x20,0x20,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0202001 <unknown>

stnt1h  {z20.h-z21.h}, pn13, [x10, x21, lsl #1]  // 10100000-00110101-00110101-01010101
// CHECK-INST: stnt1h  { z20.h, z21.h }, pn13, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x55,0x35,0x35,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0353555 <unknown>

stnt1h  {z22.h-z23.h}, pn11, [x13, x8, lsl #1]  // 10100000-00101000-00101101-10110111
// CHECK-INST: stnt1h  { z22.h, z23.h }, pn11, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb7,0x2d,0x28,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0282db7 <unknown>

stnt1h  {z30.h-z31.h}, pn15, [sp, xzr, lsl #1]  // 10100000-00111111-00111111-11111111
// CHECK-INST: stnt1h  { z30.h, z31.h }, pn15, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xff,0x3f,0x3f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a03f3fff <unknown>

stnt1h  {z0.h-z1.h}, pn8, [x0]  // 10100000-01100000-00100000-00000001
// CHECK-INST: stnt1h  { z0.h, z1.h }, pn8, [x0]
// CHECK-ENCODING: [0x01,0x20,0x60,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0602001 <unknown>

stnt1h  {z20.h-z21.h}, pn13, [x10, #10, mul vl]  // 10100000-01100101-00110101-01010101
// CHECK-INST: stnt1h  { z20.h, z21.h }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x35,0x65,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0653555 <unknown>

stnt1h  {z22.h-z23.h}, pn11, [x13, #-16, mul vl]  // 10100000-01101000-00101101-10110111
// CHECK-INST: stnt1h  { z22.h, z23.h }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x2d,0x68,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0682db7 <unknown>

stnt1h  {z30.h-z31.h}, pn15, [sp, #-2, mul vl]  // 10100000-01101111-00111111-11111111
// CHECK-INST: stnt1h  { z30.h, z31.h }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x3f,0x6f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a06f3fff <unknown>

stnt1h  {z0.h-z3.h}, pn8, [x0, x0, lsl #1]  // 10100000-00100000-10100000-00000001
// CHECK-INST: stnt1h  { z0.h - z3.h }, pn8, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x01,0xa0,0x20,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a020a001 <unknown>

stnt1h  {z20.h-z23.h}, pn13, [x10, x21, lsl #1]  // 10100000-00110101-10110101-01010101
// CHECK-INST: stnt1h  { z20.h - z23.h }, pn13, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x55,0xb5,0x35,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a035b555 <unknown>

stnt1h  {z20.h-z23.h}, pn11, [x13, x8, lsl #1]  // 10100000-00101000-10101101-10110101
// CHECK-INST: stnt1h  { z20.h - z23.h }, pn11, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb5,0xad,0x28,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a028adb5 <unknown>

stnt1h  {z28.h-z31.h}, pn15, [sp, xzr, lsl #1]  // 10100000-00111111-10111111-11111101
// CHECK-INST: stnt1h  { z28.h - z31.h }, pn15, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xfd,0xbf,0x3f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a03fbffd <unknown>

stnt1h  {z0.h-z3.h}, pn8, [x0]  // 10100000-01100000-10100000-00000001
// CHECK-INST: stnt1h  { z0.h - z3.h }, pn8, [x0]
// CHECK-ENCODING: [0x01,0xa0,0x60,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a060a001 <unknown>

stnt1h  {z20.h-z23.h}, pn13, [x10, #20, mul vl]  // 10100000-01100101-10110101-01010101
// CHECK-INST: stnt1h  { z20.h - z23.h }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x65,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a065b555 <unknown>

stnt1h  {z20.h-z23.h}, pn11, [x13, #-32, mul vl]  // 10100000-01101000-10101101-10110101
// CHECK-INST: stnt1h  { z20.h - z23.h }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb5,0xad,0x68,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a068adb5 <unknown>

stnt1h  {z28.h-z31.h}, pn15, [sp, #-4, mul vl]  // 10100000-01101111-10111111-11111101
// CHECK-INST: stnt1h  { z28.h - z31.h }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfd,0xbf,0x6f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a06fbffd <unknown>
