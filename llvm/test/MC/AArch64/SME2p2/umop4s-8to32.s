// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-mop4 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-mop4 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-mop4 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-mop4 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-mop4 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

umop4s  za0.s, z0.b, z16.b  // 10000001-00100000-10000000-00010000
// CHECK-INST: umop4s  za0.s, z0.b, z16.b
// CHECK-ENCODING: [0x10,0x80,0x20,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81208010 <unknown>

umop4s  za1.s, z10.b, z20.b  // 10000001-00100100-10000001-01010001
// CHECK-INST: umop4s  za1.s, z10.b, z20.b
// CHECK-ENCODING: [0x51,0x81,0x24,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81248151 <unknown>

umop4s  za3.s, z14.b, z30.b  // 10000001-00101110-10000001-11010011
// CHECK-INST: umop4s  za3.s, z14.b, z30.b
// CHECK-ENCODING: [0xd3,0x81,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 812e81d3 <unknown>

umop4s  za0.s, z0.b, {z16.b-z17.b}  // 10000001-00110000-10000000-00010000
// CHECK-INST: umop4s  za0.s, z0.b, { z16.b, z17.b }
// CHECK-ENCODING: [0x10,0x80,0x30,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81308010 <unknown>

umop4s  za1.s, z10.b, {z20.b-z21.b}  // 10000001-00110100-10000001-01010001
// CHECK-INST: umop4s  za1.s, z10.b, { z20.b, z21.b }
// CHECK-ENCODING: [0x51,0x81,0x34,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81348151 <unknown>

umop4s  za3.s, z14.b, {z30.b-z31.b}  // 10000001-00111110-10000001-11010011
// CHECK-INST: umop4s  za3.s, z14.b, { z30.b, z31.b }
// CHECK-ENCODING: [0xd3,0x81,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 813e81d3 <unknown>

umop4s  za0.s, {z0.b-z1.b}, z16.b  // 10000001-00100000-10000010-00010000
// CHECK-INST: umop4s  za0.s, { z0.b, z1.b }, z16.b
// CHECK-ENCODING: [0x10,0x82,0x20,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81208210 <unknown>

umop4s  za1.s, {z10.b-z11.b}, z20.b  // 10000001-00100100-10000011-01010001
// CHECK-INST: umop4s  za1.s, { z10.b, z11.b }, z20.b
// CHECK-ENCODING: [0x51,0x83,0x24,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81248351 <unknown>

umop4s  za3.s, {z14.b-z15.b}, z30.b  // 10000001-00101110-10000011-11010011
// CHECK-INST: umop4s  za3.s, { z14.b, z15.b }, z30.b
// CHECK-ENCODING: [0xd3,0x83,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 812e83d3 <unknown>

umop4s  za0.s, {z0.b-z1.b}, {z16.b-z17.b}  // 10000001-00110000-10000010-00010000
// CHECK-INST: umop4s  za0.s, { z0.b, z1.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x10,0x82,0x30,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81308210 <unknown>

umop4s  za1.s, {z10.b-z11.b}, {z20.b-z21.b}  // 10000001-00110100-10000011-01010001
// CHECK-INST: umop4s  za1.s, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x51,0x83,0x34,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81348351 <unknown>

umop4s  za3.s, {z14.b-z15.b}, {z30.b-z31.b}  // 10000001-00111110-10000011-11010011
// CHECK-INST: umop4s  za3.s, { z14.b, z15.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xd3,0x83,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 813e83d3 <unknown>
