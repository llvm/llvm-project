// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f8f32 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f8f32 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-f8f32 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f8f32 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f8f32 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-f8f32 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Single vectors

fmop4a  za0.s, z0.b, z16.b  // 10000000-00100000-00000000-00000000
// CHECK-INST: fmop4a  za0.s, z0.b, z16.b
// CHECK-ENCODING: [0x00,0x00,0x20,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80200000 <unknown>

fmop4a  za1.s, z10.b, z20.b  // 10000000-00100100-00000001-01000001
// CHECK-INST: fmop4a  za1.s, z10.b, z20.b
// CHECK-ENCODING: [0x41,0x01,0x24,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80240141 <unknown>

fmop4a  za3.s, z14.b, z30.b  // 10000000-00101110-00000001-11000011
// CHECK-INST: fmop4a  za3.s, z14.b, z30.b
// CHECK-ENCODING: [0xc3,0x01,0x2e,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 802e01c3 <unknown>

// Single and multiple vectors

fmop4a  za0.s, z0.b, {z16.b-z17.b}  // 10000000-00110000-00000000-00000000
// CHECK-INST: fmop4a  za0.s, z0.b, { z16.b, z17.b }
// CHECK-ENCODING: [0x00,0x00,0x30,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80300000 <unknown>

fmop4a  za1.s, z10.b, {z20.b-z21.b}  // 10000000-00110100-00000001-01000001
// CHECK-INST: fmop4a  za1.s, z10.b, { z20.b, z21.b }
// CHECK-ENCODING: [0x41,0x01,0x34,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80340141 <unknown>

fmop4a  za3.s, z14.b, {z30.b-z31.b}  // 10000000-00111110-00000001-11000011
// CHECK-INST: fmop4a  za3.s, z14.b, { z30.b, z31.b }
// CHECK-ENCODING: [0xc3,0x01,0x3e,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 803e01c3 <unknown>

// Multiple and single vectors

fmop4a  za0.s, {z0.b-z1.b}, z16.b  // 10000000-00100000-00000010-00000000
// CHECK-INST: fmop4a  za0.s, { z0.b, z1.b }, z16.b
// CHECK-ENCODING: [0x00,0x02,0x20,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80200200 <unknown>

fmop4a  za1.s, {z10.b-z11.b}, z20.b  // 10000000-00100100-00000011-01000001
// CHECK-INST: fmop4a  za1.s, { z10.b, z11.b }, z20.b
// CHECK-ENCODING: [0x41,0x03,0x24,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80240341 <unknown>

fmop4a  za3.s, {z14.b-z15.b}, z30.b  // 10000000-00101110-00000011-11000011
// CHECK-INST: fmop4a  za3.s, { z14.b, z15.b }, z30.b
// CHECK-ENCODING: [0xc3,0x03,0x2e,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 802e03c3 <unknown>

// Multiple vectors

fmop4a  za0.s, {z0.b-z1.b}, {z16.b-z17.b}  // 10000000-00110000-00000010-00000000
// CHECK-INST: fmop4a  za0.s, { z0.b, z1.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x00,0x02,0x30,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80300200 <unknown>

fmop4a  za1.s, {z10.b-z11.b}, {z20.b-z21.b}  // 10000000-00110100-00000011-01000001
// CHECK-INST: fmop4a  za1.s, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x41,0x03,0x34,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80340341 <unknown>

fmop4a  za3.s, {z14.b-z15.b}, {z30.b-z31.b}  // 10000000-00111110-00000011-11000011
// CHECK-INST: fmop4a  za3.s, { z14.b, z15.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc3,0x03,0x3e,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 803e03c3 <unknown>
