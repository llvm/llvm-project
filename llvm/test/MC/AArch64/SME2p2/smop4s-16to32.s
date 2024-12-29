// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

smop4s  za0.s, z0.h, z16.h  // 10000000-00000000-10000000-00011000
// CHECK-INST: smop4s  za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x18,0x80,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80008018 <unknown>

smop4s  za3.s, z12.h, z24.h  // 10000000-00001000-10000001-10011011
// CHECK-INST: smop4s  za3.s, z12.h, z24.h
// CHECK-ENCODING: [0x9b,0x81,0x08,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 8008819b <unknown>

smop4s  za3.s, z14.h, z30.h  // 10000000-00001110-10000001-11011011
// CHECK-INST: smop4s  za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xdb,0x81,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e81db <unknown>

smop4s  za0.s, z0.h, {z16.h-z17.h}  // 10000000-00010000-10000000-00011000
// CHECK-INST: smop4s  za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x80,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80108018 <unknown>

smop4s  za3.s, z12.h, {z24.h-z25.h}  // 10000000-00011000-10000001-10011011
// CHECK-INST: smop4s  za3.s, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x9b,0x81,0x18,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 8018819b <unknown>

smop4s  za3.s, z14.h, {z30.h-z31.h}  // 10000000-00011110-10000001-11011011
// CHECK-INST: smop4s  za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xdb,0x81,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e81db <unknown>

smop4s  za0.s, {z0.h-z1.h}, z16.h  // 10000000-00000000-10000010-00011000
// CHECK-INST: smop4s  za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x18,0x82,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80008218 <unknown>

smop4s  za3.s, {z12.h-z13.h}, z24.h  // 10000000-00001000-10000011-10011011
// CHECK-INST: smop4s  za3.s, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x9b,0x83,0x08,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 8008839b <unknown>

smop4s  za3.s, {z14.h-z15.h}, z30.h  // 10000000-00001110-10000011-11011011
// CHECK-INST: smop4s  za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xdb,0x83,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e83db <unknown>

smop4s  za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000000-00010000-10000010-00011000
// CHECK-INST: smop4s  za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x82,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80108218 <unknown>

smop4s  za3.s, {z12.h-z13.h}, {z24.h-z25.h}  // 10000000-00011000-10000011-10011011
// CHECK-INST: smop4s  za3.s, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x9b,0x83,0x18,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 8018839b <unknown>

smop4s  za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000000-00011110-10000011-11011011
// CHECK-INST: smop4s  za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdb,0x83,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e83db <unknown>
