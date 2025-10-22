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

usmop4a za0.s, z0.b, z16.b  // 10000001-00000000-10000000-00000000
// CHECK-INST: usmop4a za0.s, z0.b, z16.b
// CHECK-ENCODING: [0x00,0x80,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81008000 <unknown>

usmop4a za1.s, z10.b, z20.b  // 10000001-00000100-10000001-01000001
// CHECK-INST: usmop4a za1.s, z10.b, z20.b
// CHECK-ENCODING: [0x41,0x81,0x04,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81048141 <unknown>

usmop4a za3.s, z14.b, z30.b  // 10000001-00001110-10000001-11000011
// CHECK-INST: usmop4a za3.s, z14.b, z30.b
// CHECK-ENCODING: [0xc3,0x81,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e81c3 <unknown>

usmop4a za0.s, z0.b, {z16.b-z17.b}  // 10000001-00010000-10000000-00000000
// CHECK-INST: usmop4a za0.s, z0.b, { z16.b, z17.b }
// CHECK-ENCODING: [0x00,0x80,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81108000 <unknown>

usmop4a za1.s, z10.b, {z20.b-z21.b}  // 10000001-00010100-10000001-01000001
// CHECK-INST: usmop4a za1.s, z10.b, { z20.b, z21.b }
// CHECK-ENCODING: [0x41,0x81,0x14,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81148141 <unknown>

usmop4a za3.s, z14.b, {z30.b-z31.b}  // 10000001-00011110-10000001-11000011
// CHECK-INST: usmop4a za3.s, z14.b, { z30.b, z31.b }
// CHECK-ENCODING: [0xc3,0x81,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e81c3 <unknown>

usmop4a za0.s, {z0.b-z1.b}, z16.b  // 10000001-00000000-10000010-00000000
// CHECK-INST: usmop4a za0.s, { z0.b, z1.b }, z16.b
// CHECK-ENCODING: [0x00,0x82,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81008200 <unknown>

usmop4a za1.s, {z10.b-z11.b}, z20.b  // 10000001-00000100-10000011-01000001
// CHECK-INST: usmop4a za1.s, { z10.b, z11.b }, z20.b
// CHECK-ENCODING: [0x41,0x83,0x04,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81048341 <unknown>

usmop4a za3.s, {z14.b-z15.b}, z30.b  // 10000001-00001110-10000011-11000011
// CHECK-INST: usmop4a za3.s, { z14.b, z15.b }, z30.b
// CHECK-ENCODING: [0xc3,0x83,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e83c3 <unknown>

usmop4a za0.s, {z0.b-z1.b}, {z16.b-z17.b}  // 10000001-00010000-10000010-00000000
// CHECK-INST: usmop4a za0.s, { z0.b, z1.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x00,0x82,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81108200 <unknown>

usmop4a za1.s, {z10.b-z11.b}, {z20.b-z21.b}  // 10000001-00010100-10000011-01000001
// CHECK-INST: usmop4a za1.s, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x41,0x83,0x14,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81148341 <unknown>

usmop4a za3.s, {z14.b-z15.b}, {z30.b-z31.b}  // 10000001-00011110-10000011-11000011
// CHECK-INST: usmop4a za3.s, { z14.b, z15.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc3,0x83,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e83c3 <unknown>
