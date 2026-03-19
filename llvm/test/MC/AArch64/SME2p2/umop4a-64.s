// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-mop4,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-mop4,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-mop4,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-mop4 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-mop4,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

umop4a  za0.d, z0.h, z16.h  // 10100001-11100000-00000000-00001000
// CHECK-INST: umop4a  za0.d, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x00,0xe0,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1e00008 <unknown>

umop4a  za5.d, z10.h, z20.h  // 10100001-11100100-00000001-01001101
// CHECK-INST: umop4a  za5.d, z10.h, z20.h
// CHECK-ENCODING: [0x4d,0x01,0xe4,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1e4014d <unknown>

umop4a  za7.d, z14.h, z30.h  // 10100001-11101110-00000001-11001111
// CHECK-INST: umop4a  za7.d, z14.h, z30.h
// CHECK-ENCODING: [0xcf,0x01,0xee,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1ee01cf <unknown>

umop4a  za0.d, z0.h, {z16.h-z17.h}  // 10100001-11110000-00000000-00001000
// CHECK-INST: umop4a  za0.d, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x00,0xf0,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1f00008 <unknown>

umop4a  za5.d, z10.h, {z20.h-z21.h}  // 10100001-11110100-00000001-01001101
// CHECK-INST: umop4a  za5.d, z10.h, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x01,0xf4,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1f4014d <unknown>

umop4a  za7.d, z14.h, {z30.h-z31.h}  // 10100001-11111110-00000001-11001111
// CHECK-INST: umop4a  za7.d, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x01,0xfe,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1fe01cf <unknown>

umop4a  za0.d, {z0.h-z1.h}, z16.h  // 10100001-11100000-00000010-00001000
// CHECK-INST: umop4a  za0.d, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x02,0xe0,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1e00208 <unknown>

umop4a  za5.d, {z10.h-z11.h}, z20.h  // 10100001-11100100-00000011-01001101
// CHECK-INST: umop4a  za5.d, { z10.h, z11.h }, z20.h
// CHECK-ENCODING: [0x4d,0x03,0xe4,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1e4034d <unknown>

umop4a  za7.d, {z14.h-z15.h}, z30.h  // 10100001-11101110-00000011-11001111
// CHECK-INST: umop4a  za7.d, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xcf,0x03,0xee,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1ee03cf <unknown>

umop4a  za0.d, {z0.h-z1.h}, {z16.h-z17.h}  // 10100001-11110000-00000010-00001000
// CHECK-INST: umop4a  za0.d, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x02,0xf0,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1f00208 <unknown>

umop4a  za5.d, {z10.h-z11.h}, {z20.h-z21.h}  // 10100001-11110100-00000011-01001101
// CHECK-INST: umop4a  za5.d, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x03,0xf4,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1f4034d <unknown>

umop4a  za7.d, {z14.h-z15.h}, {z30.h-z31.h}  // 10100001-11111110-00000011-11001111
// CHECK-INST: umop4a  za7.d, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x03,0xfe,0xa1]
// CHECK-ERROR: instruction requires: sme-i16i64 sme-mop4
// CHECK-UNKNOWN: a1fe03cf <unknown>
