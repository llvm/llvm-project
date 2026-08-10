// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2p1 - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2p1 - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


luti2   {z0.h, z8.h}, zt0, z0[0]  // 11000000-10011100-01010000-00000000
// CHECK-INST: luti2   { z0.h, z8.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x50,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09c5000 <unknown>

luti2   {z21.h, z29.h}, zt0, z10[2]  // 11000000-10011101-01010001-01010101
// CHECK-INST: luti2   { z21.h, z29.h }, zt0, z10[2]
// CHECK-ENCODING: [0x55,0x51,0x9d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09d5155 <unknown>

luti2   {z23.h, z31.h}, zt0, z13[1]  // 11000000-10011100-11010001-10110111
// CHECK-INST: luti2   { z23.h, z31.h }, zt0, z13[1]
// CHECK-ENCODING: [0xb7,0xd1,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09cd1b7 <unknown>

luti2   {z23.h, z31.h}, zt0, z31[7]  // 11000000-10011111-11010011-11110111
// CHECK-INST: luti2   { z23.h, z31.h }, zt0, z31[7]
// CHECK-ENCODING: [0xf7,0xd3,0x9f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09fd3f7 <unknown>


luti2   {z0.b, z8.b}, zt0, z0[0]  // 11000000-10011100-01000000-00000000
// CHECK-INST: luti2   { z0.b, z8.b }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x40,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09c4000 <unknown>

luti2   {z21.b, z29.b}, zt0, z10[2]  // 11000000-10011101-01000001-01010101
// CHECK-INST: luti2   { z21.b, z29.b }, zt0, z10[2]
// CHECK-ENCODING: [0x55,0x41,0x9d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09d4155 <unknown>

luti2   {z23.b, z31.b}, zt0, z13[1]  // 11000000-10011100-11000001-10110111
// CHECK-INST: luti2   { z23.b, z31.b }, zt0, z13[1]
// CHECK-ENCODING: [0xb7,0xc1,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09cc1b7 <unknown>

luti2   {z23.b, z31.b}, zt0, z31[7]  // 11000000-10011111-11000011-11110111
// CHECK-INST: luti2   { z23.b, z31.b }, zt0, z31[7]
// CHECK-ENCODING: [0xf7,0xc3,0x9f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09fc3f7 <unknown>


luti2   {z0.h, z4.h, z8.h, z12.h}, zt0, z0[0]  // 11000000-10011100-10010000-00000000
// CHECK-INST: luti2   { z0.h, z4.h, z8.h, z12.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x90,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09c9000 <unknown>

luti2   {z17.h, z21.h, z25.h, z29.h}, zt0, z10[1]  // 11000000-10011101-10010001-01010001
// CHECK-INST: luti2   { z17.h, z21.h, z25.h, z29.h }, zt0, z10[1]
// CHECK-ENCODING: [0x51,0x91,0x9d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09d9151 <unknown>

luti2   {z19.h, z23.h, z27.h, z31.h}, zt0, z13[0]  // 11000000-10011100-10010001-10110011
// CHECK-INST: luti2   { z19.h, z23.h, z27.h, z31.h }, zt0, z13[0]
// CHECK-ENCODING: [0xb3,0x91,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09c91b3 <unknown>

luti2   {z19.h, z23.h, z27.h, z31.h}, zt0, z31[3]  // 11000000-10011111-10010011-11110011
// CHECK-INST: luti2   { z19.h, z23.h, z27.h, z31.h }, zt0, z31[3]
// CHECK-ENCODING: [0xf3,0x93,0x9f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09f93f3 <unknown>


luti2   {z0.b, z4.b, z8.b, z12.b}, zt0, z0[0]  // 11000000-10011100-10000000-00000000
// CHECK-INST: luti2   { z0.b, z4.b, z8.b, z12.b }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x80,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09c8000 <unknown>

luti2   {z17.b, z21.b, z25.b, z29.b}, zt0, z10[1]  // 11000000-10011101-10000001-01010001
// CHECK-INST: luti2   { z17.b, z21.b, z25.b, z29.b }, zt0, z10[1]
// CHECK-ENCODING: [0x51,0x81,0x9d,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09d8151 <unknown>

luti2   {z19.b, z23.b, z27.b, z31.b}, zt0, z13[0]  // 11000000-10011100-10000001-10110011
// CHECK-INST: luti2   { z19.b, z23.b, z27.b, z31.b }, zt0, z13[0]
// CHECK-ENCODING: [0xb3,0x81,0x9c,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09c81b3 <unknown>

luti2   {z19.b, z23.b, z27.b, z31.b}, zt0, z31[3]  // 11000000-10011111-10000011-11110011
// CHECK-INST: luti2   { z19.b, z23.b, z27.b, z31.b }, zt0, z31[3]
// CHECK-ENCODING: [0xf3,0x83,0x9f,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09f83f3 <unknown>

