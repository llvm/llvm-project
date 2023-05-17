// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

uqrshrn z0.h, {z0.s-z1.s}, #16  // 01000101-10110000-00111000-00000000
// CHECK-INST: uqrshrn z0.h, { z0.s, z1.s }, #16
// CHECK-ENCODING: [0x00,0x38,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45b03800 <unknown>

uqrshrn z21.h, {z10.s-z11.s}, #11  // 01000101-10110101-00111001-01010101
// CHECK-INST: uqrshrn z21.h, { z10.s, z11.s }, #11
// CHECK-ENCODING: [0x55,0x39,0xb5,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45b53955 <unknown>

uqrshrn z23.h, {z12.s-z13.s}, #8  // 01000101-10111000-00111001-10010111
// CHECK-INST: uqrshrn z23.h, { z12.s, z13.s }, #8
// CHECK-ENCODING: [0x97,0x39,0xb8,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45b83997 <unknown>

uqrshrn z31.h, {z30.s-z31.s}, #1  // 01000101-10111111-00111011-11011111
// CHECK-INST: uqrshrn z31.h, { z30.s, z31.s }, #1
// CHECK-ENCODING: [0xdf,0x3b,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45bf3bdf <unknown>
