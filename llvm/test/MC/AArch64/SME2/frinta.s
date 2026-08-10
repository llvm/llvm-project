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


frinta  {z0.s - z1.s}, {z0.s - z1.s}  // 11000001-10101100-11100000-00000000
// CHECK-INST: frinta  { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe0,0xac,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ace000 <unknown>

frinta  {z20.s - z21.s}, {z10.s - z11.s}  // 11000001-10101100-11100001-01010100
// CHECK-INST: frinta  { z20.s, z21.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x54,0xe1,0xac,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ace154 <unknown>

frinta  {z22.s - z23.s}, {z12.s - z13.s}  // 11000001-10101100-11100001-10010110
// CHECK-INST: frinta  { z22.s, z23.s }, { z12.s, z13.s }
// CHECK-ENCODING: [0x96,0xe1,0xac,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ace196 <unknown>

frinta  {z30.s - z31.s}, {z30.s - z31.s}  // 11000001-10101100-11100011-11011110
// CHECK-INST: frinta  { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xde,0xe3,0xac,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ace3de <unknown>


frinta  {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10111100-11100000-00000000
// CHECK-INST: frinta  { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0xbc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bce000 <unknown>

frinta  {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-10111100-11100001-00010100
// CHECK-INST: frinta  { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x14,0xe1,0xbc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bce114 <unknown>

frinta  {z20.s - z23.s}, {z12.s - z15.s}  // 11000001-10111100-11100001-10010100
// CHECK-INST: frinta  { z20.s - z23.s }, { z12.s - z15.s }
// CHECK-ENCODING: [0x94,0xe1,0xbc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bce194 <unknown>

frinta  {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111100-11100011-10011100
// CHECK-INST: frinta  { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0xe3,0xbc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bce39c <unknown>
