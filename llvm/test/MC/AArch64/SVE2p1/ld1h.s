// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

ld1h    {z0.h-z1.h}, pn8/z, [x0, x0, lsl #1]  // 10100000-00000000-00100000-00000000
// CHECK-INST: ld1h    { z0.h, z1.h }, pn8/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x20,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0002000 <unknown>

ld1h    {z20.h-z21.h}, pn13/z, [x10, x21, lsl #1]  // 10100000-00010101-00110101-01010100
// CHECK-INST: ld1h    { z20.h, z21.h }, pn13/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x54,0x35,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0153554 <unknown>

ld1h    {z22.h-z23.h}, pn11/z, [x13, x8, lsl #1]  // 10100000-00001000-00101101-10110110
// CHECK-INST: ld1h    { z22.h, z23.h }, pn11/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb6,0x2d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0082db6 <unknown>

ld1h    {z30.h-z31.h}, pn15/z, [sp, xzr, lsl #1]  // 10100000-00011111-00111111-11111110
// CHECK-INST: ld1h    { z30.h, z31.h }, pn15/z, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xfe,0x3f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f3ffe <unknown>

ld1h    {z0.h-z1.h}, pn8/z, [x0]  // 10100000-01000000-00100000-00000000
// CHECK-INST: ld1h    { z0.h, z1.h }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x20,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0402000 <unknown>

ld1h    {z20.h-z21.h}, pn13/z, [x10, #10, mul vl]  // 10100000-01000101-00110101-01010100
// CHECK-INST: ld1h    { z20.h, z21.h }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x54,0x35,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0453554 <unknown>

ld1h    {z22.h-z23.h}, pn11/z, [x13, #-16, mul vl]  // 10100000-01001000-00101101-10110110
// CHECK-INST: ld1h    { z22.h, z23.h }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb6,0x2d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0482db6 <unknown>

ld1h    {z30.h-z31.h}, pn15/z, [sp, #-2, mul vl]  // 10100000-01001111-00111111-11111110
// CHECK-INST: ld1h    { z30.h, z31.h }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xfe,0x3f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f3ffe <unknown>

ld1h    {z0.h-z3.h}, pn8/z, [x0, x0, lsl #1]  // 10100000-00000000-10100000-00000000
// CHECK-INST: ld1h    { z0.h - z3.h }, pn8/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a000a000 <unknown>

ld1h    {z20.h-z23.h}, pn13/z, [x10, x21, lsl #1]  // 10100000-00010101-10110101-01010100
// CHECK-INST: ld1h    { z20.h - z23.h }, pn13/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x54,0xb5,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a015b554 <unknown>

ld1h    {z20.h-z23.h}, pn11/z, [x13, x8, lsl #1]  // 10100000-00001000-10101101-10110100
// CHECK-INST: ld1h    { z20.h - z23.h }, pn11/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb4,0xad,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a008adb4 <unknown>

ld1h    {z28.h-z31.h}, pn15/z, [sp, xzr, lsl #1]  // 10100000-00011111-10111111-11111100
// CHECK-INST: ld1h    { z28.h - z31.h }, pn15/z, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xfc,0xbf,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01fbffc <unknown>

ld1h    {z0.h-z3.h}, pn8/z, [x0]  // 10100000-01000000-10100000-00000000
// CHECK-INST: ld1h    { z0.h - z3.h }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a040a000 <unknown>

ld1h    {z20.h-z23.h}, pn13/z, [x10, #20, mul vl]  // 10100000-01000101-10110101-01010100
// CHECK-INST: ld1h    { z20.h - z23.h }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x54,0xb5,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a045b554 <unknown>

ld1h    {z20.h-z23.h}, pn11/z, [x13, #-32, mul vl]  // 10100000-01001000-10101101-10110100
// CHECK-INST: ld1h    { z20.h - z23.h }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb4,0xad,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a048adb4 <unknown>

ld1h    {z28.h-z31.h}, pn15/z, [sp, #-4, mul vl]  // 10100000-01001111-10111111-11111100
// CHECK-INST: ld1h    { z28.h - z31.h }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfc,0xbf,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04fbffc <unknown>
