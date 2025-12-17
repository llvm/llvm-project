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

smop4a  za0.s, z0.h, z16.h  // 10000000-00000000-10000000-00001000
// CHECK-INST: smop4a  za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x80,0x00,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 80008008 <unknown>

smop4a  za3.s, z12.h, z24.h  // 10000000-00001000-10000001-10001011
// CHECK-INST: smop4a  za3.s, z12.h, z24.h
// CHECK-ENCODING: [0x8b,0x81,0x08,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8008818b <unknown>

smop4a  za3.s, z14.h, z30.h  // 10000000-00001110-10000001-11001011
// CHECK-INST: smop4a  za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xcb,0x81,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 800e81cb <unknown>

smop4a  za0.s, z0.h, {z16.h-z17.h}  // 10000000-00010000-10000000-00001000
// CHECK-INST: smop4a  za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x80,0x10,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 80108008 <unknown>

smop4a  za3.s, z12.h, {z24.h-z25.h}  // 10000000-00011000-10000001-10001011
// CHECK-INST: smop4a  za3.s, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x8b,0x81,0x18,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8018818b <unknown>

smop4a  za3.s, z14.h, {z30.h-z31.h}  // 10000000-00011110-10000001-11001011
// CHECK-INST: smop4a  za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xcb,0x81,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 801e81cb <unknown>

smop4a  za0.s, {z0.h-z1.h}, z16.h  // 10000000-00000000-10000010-00001000
// CHECK-INST: smop4a  za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x82,0x00,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 80008208 <unknown>

smop4a  za3.s, {z12.h-z13.h}, z24.h  // 10000000-00001000-10000011-10001011
// CHECK-INST: smop4a  za3.s, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x8b,0x83,0x08,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8008838b <unknown>

smop4a  za3.s, {z14.h-z15.h}, z30.h  // 10000000-00001110-10000011-11001011
// CHECK-INST: smop4a  za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xcb,0x83,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 800e83cb <unknown>

smop4a  za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000000-00010000-10000010-00001000
// CHECK-INST: smop4a  za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x82,0x10,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 80108208 <unknown>

smop4a  za3.s, {z12.h-z13.h}, {z24.h-z25.h}  // 10000000-00011000-10000011-10001011
// CHECK-INST: smop4a  za3.s, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x8b,0x83,0x18,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8018838b <unknown>

smop4a  za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000000-00011110-10000011-11001011
// CHECK-INST: smop4a  za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcb,0x83,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 801e83cb <unknown>
