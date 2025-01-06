// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

smop4s  za0.s, z0.b, z16.b  // 10000000-00000000-10000000-00010000
// CHECK-INST: smop4s  za0.s, z0.b, z16.b
// CHECK-ENCODING: [0x10,0x80,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80008010 <unknown>

smop4s  za1.s, z10.b, z20.b  // 10000000-00000100-10000001-01010001
// CHECK-INST: smop4s  za1.s, z10.b, z20.b
// CHECK-ENCODING: [0x51,0x81,0x04,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80048151 <unknown>

smop4s  za3.s, z14.b, z30.b  // 10000000-00001110-10000001-11010011
// CHECK-INST: smop4s  za3.s, z14.b, z30.b
// CHECK-ENCODING: [0xd3,0x81,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e81d3 <unknown>

smop4s  za0.s, z0.b, {z16.b-z17.b}  // 10000000-00010000-10000000-00010000
// CHECK-INST: smop4s  za0.s, z0.b, { z16.b, z17.b }
// CHECK-ENCODING: [0x10,0x80,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80108010 <unknown>

smop4s  za1.s, z10.b, {z20.b-z21.b}  // 10000000-00010100-10000001-01010001
// CHECK-INST: smop4s  za1.s, z10.b, { z20.b, z21.b }
// CHECK-ENCODING: [0x51,0x81,0x14,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80148151 <unknown>

smop4s  za3.s, z14.b, {z30.b-z31.b}  // 10000000-00011110-10000001-11010011
// CHECK-INST: smop4s  za3.s, z14.b, { z30.b, z31.b }
// CHECK-ENCODING: [0xd3,0x81,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e81d3 <unknown>

smop4s  za0.s, {z0.b-z1.b}, z16.b  // 10000000-00000000-10000010-00010000
// CHECK-INST: smop4s  za0.s, { z0.b, z1.b }, z16.b
// CHECK-ENCODING: [0x10,0x82,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80008210 <unknown>

smop4s  za1.s, {z10.b-z11.b}, z20.b  // 10000000-00000100-10000011-01010001
// CHECK-INST: smop4s  za1.s, { z10.b, z11.b }, z20.b
// CHECK-ENCODING: [0x51,0x83,0x04,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80048351 <unknown>

smop4s  za3.s, {z14.b-z15.b}, z30.b  // 10000000-00001110-10000011-11010011
// CHECK-INST: smop4s  za3.s, { z14.b, z15.b }, z30.b
// CHECK-ENCODING: [0xd3,0x83,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e83d3 <unknown>

smop4s  za0.s, {z0.b-z1.b}, {z16.b-z17.b}  // 10000000-00010000-10000010-00010000
// CHECK-INST: smop4s  za0.s, { z0.b, z1.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x10,0x82,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80108210 <unknown>

smop4s  za1.s, {z10.b-z11.b}, {z20.b-z21.b}  // 10000000-00010100-10000011-01010001
// CHECK-INST: smop4s  za1.s, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x51,0x83,0x14,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80148351 <unknown>

smop4s  za3.s, {z14.b-z15.b}, {z30.b-z31.b}  // 10000000-00011110-10000011-11010011
// CHECK-INST: smop4s  za3.s, { z14.b, z15.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xd3,0x83,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e83d3 <unknown>
