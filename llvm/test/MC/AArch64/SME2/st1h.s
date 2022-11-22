// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - \
// RUN:        |FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


st1h    {z0.h, z8.h}, pn8, [x0, x0, lsl #1]  // 10100001-00100000-00100000-00000000
// CHECK-INST: st1h    { z0.h, z8.h }, pn8, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x20,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1202000 <unknown>

st1h    {z21.h, z29.h}, pn13, [x10, x21, lsl #1]  // 10100001-00110101-00110101-01010101
// CHECK-INST: st1h    { z21.h, z29.h }, pn13, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x55,0x35,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1353555 <unknown>

st1h    {z23.h, z31.h}, pn11, [x13, x8, lsl #1]  // 10100001-00101000-00101101-10110111
// CHECK-INST: st1h    { z23.h, z31.h }, pn11, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb7,0x2d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1282db7 <unknown>

st1h    {z23.h, z31.h}, pn15, [sp, xzr, lsl #1]  // 10100001-00111111-00111111-11110111
// CHECK-INST: st1h    { z23.h, z31.h }, pn15, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xf7,0x3f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f3ff7 <unknown>


st1h    {z0.h, z8.h}, pn8, [x0]  // 10100001-01100000-00100000-00000000
// CHECK-INST: st1h    { z0.h, z8.h }, pn8, [x0]
// CHECK-ENCODING: [0x00,0x20,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1602000 <unknown>

st1h    {z21.h, z29.h}, pn13, [x10, #10, mul vl]  // 10100001-01100101-00110101-01010101
// CHECK-INST: st1h    { z21.h, z29.h }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x35,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1653555 <unknown>

st1h    {z23.h, z31.h}, pn11, [x13, #-16, mul vl]  // 10100001-01101000-00101101-10110111
// CHECK-INST: st1h    { z23.h, z31.h }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x2d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1682db7 <unknown>

st1h    {z23.h, z31.h}, pn15, [sp, #-2, mul vl]  // 10100001-01101111-00111111-11110111
// CHECK-INST: st1h    { z23.h, z31.h }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xf7,0x3f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f3ff7 <unknown>


st1h    {z0.h, z4.h, z8.h, z12.h}, pn8, [x0, x0, lsl #1]  // 10100001-00100000-10100000-00000000
// CHECK-INST: st1h    { z0.h, z4.h, z8.h, z12.h }, pn8, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a120a000 <unknown>

st1h    {z17.h, z21.h, z25.h, z29.h}, pn13, [x10, x21, lsl #1]  // 10100001-00110101-10110101-01010001
// CHECK-INST: st1h    { z17.h, z21.h, z25.h, z29.h }, pn13, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x51,0xb5,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a135b551 <unknown>

st1h    {z19.h, z23.h, z27.h, z31.h}, pn11, [x13, x8, lsl #1]  // 10100001-00101000-10101101-10110011
// CHECK-INST: st1h    { z19.h, z23.h, z27.h, z31.h }, pn11, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb3,0xad,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a128adb3 <unknown>

st1h    {z19.h, z23.h, z27.h, z31.h}, pn15, [sp, xzr, lsl #1]  // 10100001-00111111-10111111-11110011
// CHECK-INST: st1h    { z19.h, z23.h, z27.h, z31.h }, pn15, [sp, xzr, lsl #1]
// CHECK-ENCODING: [0xf3,0xbf,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13fbff3 <unknown>


st1h    {z0.h, z4.h, z8.h, z12.h}, pn8, [x0]  // 10100001-01100000-10100000-00000000
// CHECK-INST: st1h    { z0.h, z4.h, z8.h, z12.h }, pn8, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a160a000 <unknown>

st1h    {z17.h, z21.h, z25.h, z29.h}, pn13, [x10, #20, mul vl]  // 10100001-01100101-10110101-01010001
// CHECK-INST: st1h    { z17.h, z21.h, z25.h, z29.h }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x51,0xb5,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a165b551 <unknown>

st1h    {z19.h, z23.h, z27.h, z31.h}, pn11, [x13, #-32, mul vl]  // 10100001-01101000-10101101-10110011
// CHECK-INST: st1h    { z19.h, z23.h, z27.h, z31.h }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb3,0xad,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a168adb3 <unknown>

st1h    {z19.h, z23.h, z27.h, z31.h}, pn15, [sp, #-4, mul vl]  // 10100001-01101111-10111111-11110011
// CHECK-INST: st1h    { z19.h, z23.h, z27.h, z31.h }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xf3,0xbf,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16fbff3 <unknown>

