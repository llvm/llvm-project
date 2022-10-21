// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


fcvtn   z0.h, {z0.s, z1.s}  // 11000001-00100000-11100000-00100000
// CHECK-INST: fcvtn   z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x20,0xe0,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120e020 <unknown>

fcvtn   z21.h, {z10.s, z11.s}  // 11000001-00100000-11100001-01110101
// CHECK-INST: fcvtn   z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x75,0xe1,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120e175 <unknown>

fcvtn   z23.h, {z12.s, z13.s}  // 11000001-00100000-11100001-10110111
// CHECK-INST: fcvtn   z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0xb7,0xe1,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120e1b7 <unknown>

fcvtn   z31.h, {z30.s, z31.s}  // 11000001-00100000-11100011-11111111
// CHECK-INST: fcvtn   z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xff,0xe3,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120e3ff <unknown>

