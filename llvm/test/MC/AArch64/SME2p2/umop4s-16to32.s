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

umop4s  za0.s, z0.h, z16.h  // 10000001-00000000-10000000-00011000
// CHECK-INST: umop4s  za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x18,0x80,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81008018 <unknown>

umop4s  za3.s, z12.h, z24.h  // 10000001-00001000-10000001-10011011
// CHECK-INST: umop4s  za3.s, z12.h, z24.h
// CHECK-ENCODING: [0x9b,0x81,0x08,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8108819b <unknown>

umop4s  za3.s, z14.h, z30.h  // 10000001-00001110-10000001-11011011
// CHECK-INST: umop4s  za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xdb,0x81,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e81db <unknown>

umop4s  za0.s, z0.h, {z16.h-z17.h}  // 10000001-00010000-10000000-00011000
// CHECK-INST: umop4s  za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x80,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81108018 <unknown>

umop4s  za3.s, z12.h, {z24.h-z25.h}  // 10000001-00011000-10000001-10011011
// CHECK-INST: umop4s  za3.s, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x9b,0x81,0x18,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8118819b <unknown>

umop4s  za3.s, z14.h, {z30.h-z31.h}  // 10000001-00011110-10000001-11011011
// CHECK-INST: umop4s  za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xdb,0x81,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e81db <unknown>

umop4s  za0.s, {z0.h-z1.h}, z16.h  // 10000001-00000000-10000010-00011000
// CHECK-INST: umop4s  za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x18,0x82,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81008218 <unknown>

umop4s  za3.s, {z12.h-z13.h}, z24.h  // 10000001-00001000-10000011-10011011
// CHECK-INST: umop4s  za3.s, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x9b,0x83,0x08,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8108839b <unknown>

umop4s  za3.s, {z14.h-z15.h}, z30.h  // 10000001-00001110-10000011-11011011
// CHECK-INST: umop4s  za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xdb,0x83,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e83db <unknown>

umop4s  za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00010000-10000010-00011000
// CHECK-INST: umop4s  za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x82,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81108218 <unknown>

umop4s  za3.s, {z12.h-z13.h}, {z24.h-z25.h}  // 10000001-00011000-10000011-10011011
// CHECK-INST: umop4s  za3.s, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x9b,0x83,0x18,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8118839b <unknown>

umop4s  za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00011110-10000011-11011011
// CHECK-INST: umop4s  za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdb,0x83,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e83db <unknown>
