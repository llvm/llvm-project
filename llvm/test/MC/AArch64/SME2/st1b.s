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


st1b    {z0.b, z8.b}, pn8, [x0, x0]  // 10100001-00100000-00000000-00000000
// CHECK-INST: st1b    { z0.b, z8.b }, pn8, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1200000 <unknown>

st1b    {z21.b, z29.b}, pn13, [x10, x21]  // 10100001-00110101-00010101-01010101
// CHECK-INST: st1b    { z21.b, z29.b }, pn13, [x10, x21]
// CHECK-ENCODING: [0x55,0x15,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1351555 <unknown>

st1b    {z23.b, z31.b}, pn11, [x13, x8]  // 10100001-00101000-00001101-10110111
// CHECK-INST: st1b    { z23.b, z31.b }, pn11, [x13, x8]
// CHECK-ENCODING: [0xb7,0x0d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1280db7 <unknown>

st1b    {z23.b, z31.b}, pn15, [sp, xzr]  // 10100001-00111111-00011111-11110111
// CHECK-INST: st1b    { z23.b, z31.b }, pn15, [sp, xzr]
// CHECK-ENCODING: [0xf7,0x1f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f1ff7 <unknown>


st1b    {z0.b, z8.b}, pn8, [x0]  // 10100001-01100000-00000000-00000000
// CHECK-INST: st1b    { z0.b, z8.b }, pn8, [x0]
// CHECK-ENCODING: [0x00,0x00,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1600000 <unknown>

st1b    {z21.b, z29.b}, pn13, [x10, #10, mul vl]  // 10100001-01100101-00010101-01010101
// CHECK-INST: st1b    { z21.b, z29.b }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x15,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1651555 <unknown>

st1b    {z23.b, z31.b}, pn11, [x13, #-16, mul vl]  // 10100001-01101000-00001101-10110111
// CHECK-INST: st1b    { z23.b, z31.b }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x0d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1680db7 <unknown>

st1b    {z23.b, z31.b}, pn15, [sp, #-2, mul vl]  // 10100001-01101111-00011111-11110111
// CHECK-INST: st1b    { z23.b, z31.b }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xf7,0x1f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f1ff7 <unknown>


st1b    {z0.b, z4.b, z8.b, z12.b}, pn8, [x0, x0]  // 10100001-00100000-10000000-00000000
// CHECK-INST: st1b    { z0.b, z4.b, z8.b, z12.b }, pn8, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x20,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1208000 <unknown>

st1b    {z17.b, z21.b, z25.b, z29.b}, pn13, [x10, x21]  // 10100001-00110101-10010101-01010001
// CHECK-INST: st1b    { z17.b, z21.b, z25.b, z29.b }, pn13, [x10, x21]
// CHECK-ENCODING: [0x51,0x95,0x35,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1359551 <unknown>

st1b    {z19.b, z23.b, z27.b, z31.b}, pn11, [x13, x8]  // 10100001-00101000-10001101-10110011
// CHECK-INST: st1b    { z19.b, z23.b, z27.b, z31.b }, pn11, [x13, x8]
// CHECK-ENCODING: [0xb3,0x8d,0x28,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1288db3 <unknown>

st1b    {z19.b, z23.b, z27.b, z31.b}, pn15, [sp, xzr]  // 10100001-00111111-10011111-11110011
// CHECK-INST: st1b    { z19.b, z23.b, z27.b, z31.b }, pn15, [sp, xzr]
// CHECK-ENCODING: [0xf3,0x9f,0x3f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a13f9ff3 <unknown>


st1b    {z0.b, z4.b, z8.b, z12.b}, pn8, [x0]  // 10100001-01100000-10000000-00000000
// CHECK-INST: st1b    { z0.b, z4.b, z8.b, z12.b }, pn8, [x0]
// CHECK-ENCODING: [0x00,0x80,0x60,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1608000 <unknown>

st1b    {z17.b, z21.b, z25.b, z29.b}, pn13, [x10, #20, mul vl]  // 10100001-01100101-10010101-01010001
// CHECK-INST: st1b    { z17.b, z21.b, z25.b, z29.b }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x51,0x95,0x65,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1659551 <unknown>

st1b    {z19.b, z23.b, z27.b, z31.b}, pn11, [x13, #-32, mul vl]  // 10100001-01101000-10001101-10110011
// CHECK-INST: st1b    { z19.b, z23.b, z27.b, z31.b }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb3,0x8d,0x68,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1688db3 <unknown>

st1b    {z19.b, z23.b, z27.b, z31.b}, pn15, [sp, #-4, mul vl]  // 10100001-01101111-10011111-11110011
// CHECK-INST: st1b    { z19.b, z23.b, z27.b, z31.b }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xf3,0x9f,0x6f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a16f9ff3 <unknown>

