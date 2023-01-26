// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p1,+sme-f16f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1,+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fcvtl   {z0.s, z1.s}, z0.h  // 11000001-10100000-11100000-00000001
// CHECK-INST: fcvtl   { z0.s, z1.s }, z0.h
// CHECK-ENCODING: [0x01,0xe0,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a0e001 <unknown>

fcvtl   {z20.s, z21.s}, z10.h  // 11000001-10100000-11100001-01010101
// CHECK-INST: fcvtl   { z20.s, z21.s }, z10.h
// CHECK-ENCODING: [0x55,0xe1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a0e155 <unknown>

fcvtl   {z22.s, z23.s}, z13.h  // 11000001-10100000-11100001-10110111
// CHECK-INST: fcvtl   { z22.s, z23.s }, z13.h
// CHECK-ENCODING: [0xb7,0xe1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a0e1b7 <unknown>

fcvtl   {z30.s, z31.s}, z31.h  // 11000001-10100000-11100011-11111111
// CHECK-INST: fcvtl   { z30.s, z31.s }, z31.h
// CHECK-ENCODING: [0xff,0xe3,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p1 sme-f16f16
// CHECK-UNKNOWN: c1a0e3ff <unknown>
