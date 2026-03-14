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


sqcvtu  z0.h, {z0.s, z1.s}  // 11000001-01100011-11100000-00000000
// CHECK-INST: sqcvtu  z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe0,0x63,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c163e000 <unknown>

sqcvtu  z21.h, {z10.s, z11.s}  // 11000001-01100011-11100001-01010101
// CHECK-INST: sqcvtu  z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x55,0xe1,0x63,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c163e155 <unknown>

sqcvtu  z23.h, {z12.s, z13.s}  // 11000001-01100011-11100001-10010111
// CHECK-INST: sqcvtu  z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0x97,0xe1,0x63,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c163e197 <unknown>

sqcvtu  z31.h, {z30.s, z31.s}  // 11000001-01100011-11100011-11011111
// CHECK-INST: sqcvtu  z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xdf,0xe3,0x63,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c163e3df <unknown>


sqcvtu  z0.b, {z0.s - z3.s}  // 11000001-01110011-11100000-00000000
// CHECK-INST: sqcvtu  z0.b, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0x73,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c173e000 <unknown>

sqcvtu  z21.b, {z8.s - z11.s}  // 11000001-01110011-11100001-00010101
// CHECK-INST: sqcvtu  z21.b, { z8.s - z11.s }
// CHECK-ENCODING: [0x15,0xe1,0x73,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c173e115 <unknown>

sqcvtu  z23.b, {z12.s - z15.s}  // 11000001-01110011-11100001-10010111
// CHECK-INST: sqcvtu  z23.b, { z12.s - z15.s }
// CHECK-ENCODING: [0x97,0xe1,0x73,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c173e197 <unknown>

sqcvtu  z31.b, {z28.s - z31.s}  // 11000001-01110011-11100011-10011111
// CHECK-INST: sqcvtu  z31.b, { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0xe3,0x73,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c173e39f <unknown>


sqcvtu  z0.h, {z0.d - z3.d}  // 11000001-11110011-11100000-00000000
// CHECK-INST: sqcvtu  z0.h, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0xe0,0xf3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f3e000 <unknown>

sqcvtu  z21.h, {z8.d - z11.d}  // 11000001-11110011-11100001-00010101
// CHECK-INST: sqcvtu  z21.h, { z8.d - z11.d }
// CHECK-ENCODING: [0x15,0xe1,0xf3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f3e115 <unknown>

sqcvtu  z23.h, {z12.d - z15.d}  // 11000001-11110011-11100001-10010111
// CHECK-INST: sqcvtu  z23.h, { z12.d - z15.d }
// CHECK-ENCODING: [0x97,0xe1,0xf3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f3e197 <unknown>

sqcvtu  z31.h, {z28.d - z31.d}  // 11000001-11110011-11100011-10011111
// CHECK-INST: sqcvtu  z31.h, { z28.d - z31.d }
// CHECK-ENCODING: [0x9f,0xe3,0xf3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f3e39f <unknown>

