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


sqcvt   z0.h, {z0.s, z1.s}  // 11000001-00100011-11100000-00000000
// CHECK-INST: sqcvt   z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe0,0x23,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c123e000 <unknown>

sqcvt   z21.h, {z10.s, z11.s}  // 11000001-00100011-11100001-01010101
// CHECK-INST: sqcvt   z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0xe1,0x23,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c123e155 <unknown>

sqcvt   z23.h, {z12.s, z13.s}  // 11000001-00100011-11100001-10010111
// CHECK-INST: sqcvt   z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0xe1,0x23,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c123e197 <unknown>

sqcvt   z31.h, {z30.s, z31.s}  // 11000001-00100011-11100011-11011111
// CHECK-INST: sqcvt   z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0xe3,0x23,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c123e3df <unknown>


sqcvt   z0.b, {z0.s - z3.s}  // 11000001-00110011-11100000-00000000
// CHECK-INST: sqcvt   z0.b, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e000 <unknown>

sqcvt   z21.b, {z8.s - z11.s}  // 11000001-00110011-11100001-00010101
// CHECK-INST: sqcvt   z21.b, { z8.s - z11.s }
// CHECK-ENCODING: [0x15,0xe1,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e115 <unknown>

sqcvt   z23.b, {z12.s - z15.s}  // 11000001-00110011-11100001-10010111
// CHECK-INST: sqcvt   z23.b, { z12.s - z15.s }
// CHECK-ENCODING: [0x97,0xe1,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e197 <unknown>

sqcvt   z31.b, {z28.s - z31.s}  // 11000001-00110011-11100011-10011111
// CHECK-INST: sqcvt   z31.b, { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0xe3,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e39f <unknown>


sqcvt   z0.h, {z0.d - z3.d}  // 11000001-10110011-11100000-00000000
// CHECK-INST: sqcvt   z0.h, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0xe0,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e000 <unknown>

sqcvt   z21.h, {z8.d - z11.d}  // 11000001-10110011-11100001-00010101
// CHECK-INST: sqcvt   z21.h, { z8.d - z11.d }
// CHECK-ENCODING: [0x15,0xe1,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e115 <unknown>

sqcvt   z23.h, {z12.d - z15.d}  // 11000001-10110011-11100001-10010111
// CHECK-INST: sqcvt   z23.h, { z12.d - z15.d }
// CHECK-ENCODING: [0x97,0xe1,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e197 <unknown>

sqcvt   z31.h, {z28.d - z31.d}  // 11000001-10110011-11100011-10011111
// CHECK-INST: sqcvt   z31.h, { z28.d - z31.d }
// CHECK-ENCODING: [0x9f,0xe3,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e39f <unknown>

