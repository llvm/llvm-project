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


uqcvtn  z0.b, {z0.s - z3.s}  // 11000001-00110011-11100000-01100000
// CHECK-INST: uqcvtn  z0.b, { z0.s - z3.s }
// CHECK-ENCODING: [0x60,0xe0,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e060 <unknown>

uqcvtn  z21.b, {z8.s - z11.s}  // 11000001-00110011-11100001-01110101
// CHECK-INST: uqcvtn  z21.b, { z8.s - z11.s }
// CHECK-ENCODING: [0x75,0xe1,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e175 <unknown>

uqcvtn  z23.b, {z12.s - z15.s}  // 11000001-00110011-11100001-11110111
// CHECK-INST: uqcvtn  z23.b, { z12.s - z15.s }
// CHECK-ENCODING: [0xf7,0xe1,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e1f7 <unknown>

uqcvtn  z31.b, {z28.s - z31.s}  // 11000001-00110011-11100011-11111111
// CHECK-INST: uqcvtn  z31.b, { z28.s - z31.s }
// CHECK-ENCODING: [0xff,0xe3,0x33,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c133e3ff <unknown>


uqcvtn  z0.h, {z0.d - z3.d}  // 11000001-10110011-11100000-01100000
// CHECK-INST: uqcvtn  z0.h, { z0.d - z3.d }
// CHECK-ENCODING: [0x60,0xe0,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e060 <unknown>

uqcvtn  z21.h, {z8.d - z11.d}  // 11000001-10110011-11100001-01110101
// CHECK-INST: uqcvtn  z21.h, { z8.d - z11.d }
// CHECK-ENCODING: [0x75,0xe1,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e175 <unknown>

uqcvtn  z23.h, {z12.d - z15.d}  // 11000001-10110011-11100001-11110111
// CHECK-INST: uqcvtn  z23.h, { z12.d - z15.d }
// CHECK-ENCODING: [0xf7,0xe1,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e1f7 <unknown>

uqcvtn  z31.h, {z28.d - z31.d}  // 11000001-10110011-11100011-11111111
// CHECK-INST: uqcvtn  z31.h, { z28.d - z31.d }
// CHECK-ENCODING: [0xff,0xe3,0xb3,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b3e3ff <unknown>

