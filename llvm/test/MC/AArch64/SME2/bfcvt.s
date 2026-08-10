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


bfcvt   z0.h, {z0.s, z1.s}  // 11000001-01100000-11100000-00000000
// CHECK-INST: bfcvt   z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e000 <unknown>

bfcvt   z21.h, {z10.s, z11.s}  // 11000001-01100000-11100001-01010101
// CHECK-INST: bfcvt   z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0xe1,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e155 <unknown>

bfcvt   z23.h, {z12.s, z13.s}  // 11000001-01100000-11100001-10010111
// CHECK-INST: bfcvt   z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0xe1,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e197 <unknown>

bfcvt   z31.h, {z30.s, z31.s}  // 11000001-01100000-11100011-11011111
// CHECK-INST: bfcvt   z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0xe3,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e3df <unknown>

