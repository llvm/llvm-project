// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


scvtf   {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-00100010-11100000-00000000
// CHECK-INST: scvtf   { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe0,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c122e000 <unknown>

scvtf   {z20.s, z21.s}, {z10.s, z11.s}  // 11000001-00100010-11100001-01010100
// CHECK-INST: scvtf   { z20.s, z21.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x54,0xe1,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c122e154 <unknown>

scvtf   {z22.s, z23.s}, {z12.s, z13.s}  // 11000001-00100010-11100001-10010110
// CHECK-INST: scvtf   { z22.s, z23.s }, { z12.s, z13.s }
// CHECK-ENCODING: [0x96,0xe1,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c122e196 <unknown>

scvtf   {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-00100010-11100011-11011110
// CHECK-INST: scvtf   { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xde,0xe3,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c122e3de <unknown>


scvtf   {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-00110010-11100000-00000000
// CHECK-INST: scvtf   { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c132e000 <unknown>

scvtf   {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-00110010-11100001-00010100
// CHECK-INST: scvtf   { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x14,0xe1,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c132e114 <unknown>

scvtf   {z20.s - z23.s}, {z12.s - z15.s}  // 11000001-00110010-11100001-10010100
// CHECK-INST: scvtf   { z20.s - z23.s }, { z12.s - z15.s }
// CHECK-ENCODING: [0x94,0xe1,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c132e194 <unknown>

scvtf   {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-00110010-11100011-10011100
// CHECK-INST: scvtf   { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0xe3,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c132e39c <unknown>

