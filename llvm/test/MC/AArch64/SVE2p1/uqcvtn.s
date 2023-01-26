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

uqcvtn  z0.h, {z0.s-z1.s}  // 01000101-00110001-01001000-00000000
// CHECK-INST: uqcvtn  z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x48,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45314800 <unknown>

uqcvtn  z21.h, {z10.s-z11.s}  // 01000101-00110001-01001001-01010101
// CHECK-INST: uqcvtn  z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0x49,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45314955 <unknown>

uqcvtn  z23.h, {z12.s-z13.s}  // 01000101-00110001-01001001-10010111
// CHECK-INST: uqcvtn  z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0x49,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45314997 <unknown>

uqcvtn  z31.h, {z30.s-z31.s}  // 01000101-00110001-01001011-11011111
// CHECK-INST: uqcvtn  z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0x4b,0x31,0x45]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 45314bdf <unknown>
