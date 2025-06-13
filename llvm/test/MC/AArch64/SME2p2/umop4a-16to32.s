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


umop4a  za0.s, z0.h, z16.h  // 10000001-00000000-10000000-00001000
// CHECK-INST: umop4a  za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x80,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81008008 <unknown>

umop4a  za3.s, z12.h, z24.h  // 10000001-00001000-10000001-10001011
// CHECK-INST: umop4a  za3.s, z12.h, z24.h
// CHECK-ENCODING: [0x8b,0x81,0x08,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8108818b <unknown>

umop4a  za3.s, z14.h, z30.h  // 10000001-00001110-10000001-11001011
// CHECK-INST: umop4a  za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xcb,0x81,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e81cb <unknown>

umop4a  za0.s, z0.h, {z16.h-z17.h}  // 10000001-00010000-10000000-00001000
// CHECK-INST: umop4a  za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x80,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81108008 <unknown>

umop4a  za3.s, z12.h, {z24.h-z25.h}  // 10000001-00011000-10000001-10001011
// CHECK-INST: umop4a  za3.s, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x8b,0x81,0x18,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8118818b <unknown>

umop4a  za3.s, z14.h, {z30.h-z31.h}  // 10000001-00011110-10000001-11001011
// CHECK-INST: umop4a  za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xcb,0x81,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e81cb <unknown>

umop4a  za0.s, {z0.h-z1.h}, z16.h  // 10000001-00000000-10000010-00001000
// CHECK-INST: umop4a  za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x82,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81008208 <unknown>

umop4a  za3.s, {z12.h-z13.h}, z24.h  // 10000001-00001000-10000011-10001011
// CHECK-INST: umop4a  za3.s, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x8b,0x83,0x08,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8108838b <unknown>

umop4a  za3.s, {z14.h-z15.h}, z30.h  // 10000001-00001110-10000011-11001011
// CHECK-INST: umop4a  za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xcb,0x83,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e83cb <unknown>

umop4a  za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00010000-10000010-00001000
// CHECK-INST: umop4a  za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x82,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81108208 <unknown>

umop4a  za3.s, {z12.h-z13.h}, {z24.h-z25.h}  // 10000001-00011000-10000011-10001011
// CHECK-INST: umop4a  za3.s, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x8b,0x83,0x18,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 8118838b <unknown>

umop4a  za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00011110-10000011-11001011
// CHECK-INST: umop4a  za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcb,0x83,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e83cb <unknown>
