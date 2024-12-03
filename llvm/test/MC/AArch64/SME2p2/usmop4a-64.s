// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

usmop4a za0.d, z0.h, z16.h  // 10100001-11000000-00000000-00001000
// CHECK-INST: usmop4a za0.d, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x00,0xc0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c00008 <unknown>

usmop4a za5.d, z10.h, z20.h  // 10100001-11000100-00000001-01001101
// CHECK-INST: usmop4a za5.d, z10.h, z20.h
// CHECK-ENCODING: [0x4d,0x01,0xc4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c4014d <unknown>

usmop4a za7.d, z14.h, z30.h  // 10100001-11001110-00000001-11001111
// CHECK-INST: usmop4a za7.d, z14.h, z30.h
// CHECK-ENCODING: [0xcf,0x01,0xce,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1ce01cf <unknown>

usmop4a za0.d, z0.h, {z16.h-z17.h}  // 10100001-11010000-00000000-00001000
// CHECK-INST: usmop4a za0.d, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x00,0xd0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d00008 <unknown>

usmop4a za5.d, z10.h, {z20.h-z21.h}  // 10100001-11010100-00000001-01001101
// CHECK-INST: usmop4a za5.d, z10.h, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x01,0xd4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d4014d <unknown>

usmop4a za7.d, z14.h, {z30.h-z31.h}  // 10100001-11011110-00000001-11001111
// CHECK-INST: usmop4a za7.d, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x01,0xde,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1de01cf <unknown>

usmop4a za0.d, {z0.h-z1.h}, z16.h  // 10100001-11000000-00000010-00001000
// CHECK-INST: usmop4a za0.d, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x02,0xc0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c00208 <unknown>

usmop4a za5.d, {z10.h-z11.h}, z20.h  // 10100001-11000100-00000011-01001101
// CHECK-INST: usmop4a za5.d, { z10.h, z11.h }, z20.h
// CHECK-ENCODING: [0x4d,0x03,0xc4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c4034d <unknown>

usmop4a za7.d, {z14.h-z15.h}, z30.h  // 10100001-11001110-00000011-11001111
// CHECK-INST: usmop4a za7.d, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xcf,0x03,0xce,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1ce03cf <unknown>

usmop4a za0.d, {z0.h-z1.h}, {z16.h-z17.h}  // 10100001-11010000-00000010-00001000
// CHECK-INST: usmop4a za0.d, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x02,0xd0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d00208 <unknown>

usmop4a za5.d, {z10.h-z11.h}, {z20.h-z21.h}  // 10100001-11010100-00000011-01001101
// CHECK-INST: usmop4a za5.d, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x4d,0x03,0xd4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d4034d <unknown>

usmop4a za7.d, {z14.h-z15.h}, {z30.h-z31.h}  // 10100001-11011110-00000011-11001111
// CHECK-INST: usmop4a za7.d, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x03,0xde,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1de03cf <unknown>
