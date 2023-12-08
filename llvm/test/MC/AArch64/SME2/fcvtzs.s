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


fcvtzs  {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-00100001-11100000-00000000
// CHECK-INST: fcvtzs  { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe0,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121e000 <unknown>

fcvtzs  {z20.s, z21.s}, {z10.s, z11.s}  // 11000001-00100001-11100001-01010100
// CHECK-INST: fcvtzs  { z20.s, z21.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x54,0xe1,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121e154 <unknown>

fcvtzs  {z22.s, z23.s}, {z12.s, z13.s}  // 11000001-00100001-11100001-10010110
// CHECK-INST: fcvtzs  { z22.s, z23.s }, { z12.s, z13.s }
// CHECK-ENCODING: [0x96,0xe1,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121e196 <unknown>

fcvtzs  {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-00100001-11100011-11011110
// CHECK-INST: fcvtzs  { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xde,0xe3,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121e3de <unknown>


fcvtzs  {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-00110001-11100000-00000000
// CHECK-INST: fcvtzs  { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131e000 <unknown>

fcvtzs  {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-00110001-11100001-00010100
// CHECK-INST: fcvtzs  { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x14,0xe1,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131e114 <unknown>

fcvtzs  {z20.s - z23.s}, {z12.s - z15.s}  // 11000001-00110001-11100001-10010100
// CHECK-INST: fcvtzs  { z20.s - z23.s }, { z12.s - z15.s }
// CHECK-ENCODING: [0x94,0xe1,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131e194 <unknown>

fcvtzs  {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-00110001-11100011-10011100
// CHECK-INST: fcvtzs  { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0xe3,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c131e39c <unknown>

