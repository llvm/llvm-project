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

usmop4s za0.d, z0.h, z16.h  // 10100001-11000000-00000000-00011000
// CHECK-INST: usmop4s za0.d, z0.h, z16.h
// CHECK-ENCODING: [0x18,0x00,0xc0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c00018 <unknown>

usmop4s za5.d, z10.h, z20.h  // 10100001-11000100-00000001-01011101
// CHECK-INST: usmop4s za5.d, z10.h, z20.h
// CHECK-ENCODING: [0x5d,0x01,0xc4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c4015d <unknown>

usmop4s za7.d, z14.h, z30.h  // 10100001-11001110-00000001-11011111
// CHECK-INST: usmop4s za7.d, z14.h, z30.h
// CHECK-ENCODING: [0xdf,0x01,0xce,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1ce01df <unknown>

usmop4s za0.d, z0.h, {z16.h-z17.h}  // 10100001-11010000-00000000-00011000
// CHECK-INST: usmop4s za0.d, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x00,0xd0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d00018 <unknown>

usmop4s za5.d, z10.h, {z20.h-z21.h}  // 10100001-11010100-00000001-01011101
// CHECK-INST: usmop4s za5.d, z10.h, { z20.h, z21.h }
// CHECK-ENCODING: [0x5d,0x01,0xd4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d4015d <unknown>

usmop4s za7.d, z14.h, {z30.h-z31.h}  // 10100001-11011110-00000001-11011111
// CHECK-INST: usmop4s za7.d, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x01,0xde,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1de01df <unknown>

usmop4s za0.d, {z0.h-z1.h}, z16.h  // 10100001-11000000-00000010-00011000
// CHECK-INST: usmop4s za0.d, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x18,0x02,0xc0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c00218 <unknown>

usmop4s za5.d, {z10.h-z11.h}, z20.h  // 10100001-11000100-00000011-01011101
// CHECK-INST: usmop4s za5.d, { z10.h, z11.h }, z20.h
// CHECK-ENCODING: [0x5d,0x03,0xc4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1c4035d <unknown>

usmop4s za7.d, {z14.h-z15.h}, z30.h  // 10100001-11001110-00000011-11011111
// CHECK-INST: usmop4s za7.d, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xdf,0x03,0xce,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1ce03df <unknown>

usmop4s za0.d, {z0.h-z1.h}, {z16.h-z17.h}  // 10100001-11010000-00000010-00011000
// CHECK-INST: usmop4s za0.d, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x02,0xd0,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d00218 <unknown>

usmop4s za5.d, {z10.h-z11.h}, {z20.h-z21.h}  // 10100001-11010100-00000011-01011101
// CHECK-INST: usmop4s za5.d, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x5d,0x03,0xd4,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1d4035d <unknown>

usmop4s za7.d, {z14.h-z15.h}, {z30.h-z31.h}  // 10100001-11011110-00000011-11011111
// CHECK-INST: usmop4s za7.d, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0x03,0xde,0xa1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: a1de03df <unknown>
