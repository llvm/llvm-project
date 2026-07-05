// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sqcvtun z0.h, {z0.s-z1.s}  // 01000101-00110001-01010000-00000000
// CHECK-INST: sqcvtun z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x50,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45315000 <unknown>

sqcvtun z21.h, {z10.s-z11.s}  // 01000101-00110001-01010001-01010101
// CHECK-INST: sqcvtun z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0x51,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45315155 <unknown>

sqcvtun z23.h, {z12.s-z13.s}  // 01000101-00110001-01010001-10010111
// CHECK-INST: sqcvtun z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0x51,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45315197 <unknown>

sqcvtun z31.h, {z30.s-z31.s}  // 01000101-00110001-01010011-11011111
// CHECK-INST: sqcvtun z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x53,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 453153df <unknown>
