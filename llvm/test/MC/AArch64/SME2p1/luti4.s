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


luti4   {z0.h, z8.h}, zt0, z0[0]  // 11000000-10011010-01010000-00000000
// CHECK-INST: luti4   { z0.h, z8.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x50,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09a5000 <unknown>

luti4   {z21.h, z29.h}, zt0, z10[2]  // 11000000-10011011-01010001-01010101
// CHECK-INST: luti4   { z21.h, z29.h }, zt0, z10[2]
// CHECK-ENCODING: [0x55,0x51,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09b5155 <unknown>

luti4   {z23.h, z31.h}, zt0, z13[1]  // 11000000-10011010-11010001-10110111
// CHECK-INST: luti4   { z23.h, z31.h }, zt0, z13[1]
// CHECK-ENCODING: [0xb7,0xd1,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09ad1b7 <unknown>

luti4   {z23.h, z31.h}, zt0, z31[3]  // 11000000-10011011-11010011-11110111
// CHECK-INST: luti4   { z23.h, z31.h }, zt0, z31[3]
// CHECK-ENCODING: [0xf7,0xd3,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09bd3f7 <unknown>


luti4   {z0.b, z8.b}, zt0, z0[0]  // 11000000-10011010-01000000-00000000
// CHECK-INST: luti4   { z0.b, z8.b }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x40,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09a4000 <unknown>

luti4   {z21.b, z29.b}, zt0, z10[2]  // 11000000-10011011-01000001-01010101
// CHECK-INST: luti4   { z21.b, z29.b }, zt0, z10[2]
// CHECK-ENCODING: [0x55,0x41,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09b4155 <unknown>

luti4   {z23.b, z31.b}, zt0, z13[1]  // 11000000-10011010-11000001-10110111
// CHECK-INST: luti4   { z23.b, z31.b }, zt0, z13[1]
// CHECK-ENCODING: [0xb7,0xc1,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09ac1b7 <unknown>

luti4   {z23.b, z31.b}, zt0, z31[3]  // 11000000-10011011-11000011-11110111
// CHECK-INST: luti4   { z23.b, z31.b }, zt0, z31[3]
// CHECK-ENCODING: [0xf7,0xc3,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09bc3f7 <unknown>


luti4   {z0.h, z4.h, z8.h, z12.h}, zt0, z0[0]  // 11000000-10011010-10010000-00000000
// CHECK-INST: luti4   { z0.h, z4.h, z8.h, z12.h }, zt0, z0[0]
// CHECK-ENCODING: [0x00,0x90,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09a9000 <unknown>

luti4   {z17.h, z21.h, z25.h, z29.h}, zt0, z10[1]  // 11000000-10011011-10010001-01010001
// CHECK-INST: luti4   { z17.h, z21.h, z25.h, z29.h }, zt0, z10[1]
// CHECK-ENCODING: [0x51,0x91,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09b9151 <unknown>

luti4   {z19.h, z23.h, z27.h, z31.h}, zt0, z13[0]  // 11000000-10011010-10010001-10110011
// CHECK-INST: luti4   { z19.h, z23.h, z27.h, z31.h }, zt0, z13[0]
// CHECK-ENCODING: [0xb3,0x91,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09a91b3 <unknown>

luti4   {z19.h, z23.h, z27.h, z31.h}, zt0, z31[1]  // 11000000-10011011-10010011-11110011
// CHECK-INST: luti4   { z19.h, z23.h, z27.h, z31.h }, zt0, z31[1]
// CHECK-ENCODING: [0xf3,0x93,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c09b93f3 <unknown>
