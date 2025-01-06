
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f64f64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-f64f64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f64f64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-f64f64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// FMOP4A

// Single vectors

fmop4a  za0.d, z0.d, z16.d  // 10000000-11000000-00000000-00001000
// CHECK-INST: fmop4a  za0.d, z0.d, z16.d
// CHECK-ENCODING: [0x08,0x00,0xc0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c00008 <unknown>

fmop4a  za5.d, z10.d, z20.d  // 10000000-11000100-00000001-01001101
// CHECK-INST: fmop4a  za5.d, z10.d, z20.d
// CHECK-ENCODING: [0x4d,0x01,0xc4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c4014d <unknown>

fmop4a  za7.d, z14.d, z30.d  // 10000000-11001110-00000001-11001111
// CHECK-INST: fmop4a  za7.d, z14.d, z30.d
// CHECK-ENCODING: [0xcf,0x01,0xce,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80ce01cf <unknown>

// Single and multiple vectors

fmop4a  za0.d, z0.d, {z16.d-z17.d}  // 10000000-11010000-00000000-00001000
// CHECK-INST: fmop4a  za0.d, z0.d, { z16.d, z17.d }
// CHECK-ENCODING: [0x08,0x00,0xd0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d00008 <unknown>

fmop4a  za5.d, z10.d, {z20.d-z21.d}  // 10000000-11010100-00000001-01001101
// CHECK-INST: fmop4a  za5.d, z10.d, { z20.d, z21.d }
// CHECK-ENCODING: [0x4d,0x01,0xd4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d4014d <unknown>

fmop4a  za7.d, z14.d, {z30.d-z31.d}  // 10000000-11011110-00000001-11001111
// CHECK-INST: fmop4a  za7.d, z14.d, { z30.d, z31.d }
// CHECK-ENCODING: [0xcf,0x01,0xde,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80de01cf <unknown>

// Multiple and single vectors

fmop4a  za0.d, {z0.d-z1.d}, z16.d  // 10000000-11000000-00000010-00001000
// CHECK-INST: fmop4a  za0.d, { z0.d, z1.d }, z16.d
// CHECK-ENCODING: [0x08,0x02,0xc0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c00208 <unknown>

fmop4a  za5.d, {z10.d-z11.d}, z20.d  // 10000000-11000100-00000011-01001101
// CHECK-INST: fmop4a  za5.d, { z10.d, z11.d }, z20.d
// CHECK-ENCODING: [0x4d,0x03,0xc4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c4034d <unknown>

fmop4a  za7.d, {z14.d-z15.d}, z30.d  // 10000000-11001110-00000011-11001111
// CHECK-INST: fmop4a  za7.d, { z14.d, z15.d }, z30.d
// CHECK-ENCODING: [0xcf,0x03,0xce,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80ce03cf <unknown>

// Multiple vectors

fmop4a  za0.d, {z0.d-z1.d}, {z16.d-z17.d}  // 10000000-11010000-00000010-00001000
// CHECK-INST: fmop4a  za0.d, { z0.d, z1.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x08,0x02,0xd0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d00208 <unknown>

fmop4a  za5.d, {z10.d-z11.d}, {z20.d-z21.d}  // 10000000-11010100-00000011-01001101
// CHECK-INST: fmop4a  za5.d, { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x4d,0x03,0xd4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d4034d <unknown>

fmop4a  za7.d, {z14.d-z15.d}, {z30.d-z31.d}  // 10000000-11011110-00000011-11001111
// CHECK-INST: fmop4a  za7.d, { z14.d, z15.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xcf,0x03,0xde,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80de03cf <unknown>


// FMOP4S

// Single vectors

fmop4s  za0.d, z0.d, z16.d  // 10000000-11000000-00000000-00011000
// CHECK-INST: fmop4s  za0.d, z0.d, z16.d
// CHECK-ENCODING: [0x18,0x00,0xc0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c00018 <unknown>

fmop4s  za5.d, z10.d, z20.d  // 10000000-11000100-00000001-01011101
// CHECK-INST: fmop4s  za5.d, z10.d, z20.d
// CHECK-ENCODING: [0x5d,0x01,0xc4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c4015d <unknown>

fmop4s  za7.d, z14.d, z30.d  // 10000000-11001110-00000001-11011111
// CHECK-INST: fmop4s  za7.d, z14.d, z30.d
// CHECK-ENCODING: [0xdf,0x01,0xce,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80ce01df <unknown>

// Single and multiple vectors

fmop4s  za0.d, z0.d, {z16.d-z17.d}  // 10000000-11010000-00000000-00011000
// CHECK-INST: fmop4s  za0.d, z0.d, { z16.d, z17.d }
// CHECK-ENCODING: [0x18,0x00,0xd0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d00018 <unknown>

fmop4s  za5.d, z10.d, {z20.d-z21.d}  // 10000000-11010100-00000001-01011101
// CHECK-INST: fmop4s  za5.d, z10.d, { z20.d, z21.d }
// CHECK-ENCODING: [0x5d,0x01,0xd4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d4015d <unknown>

fmop4s  za7.d, z14.d, {z30.d-z31.d}  // 10000000-11011110-00000001-11011111
// CHECK-INST: fmop4s  za7.d, z14.d, { z30.d, z31.d }
// CHECK-ENCODING: [0xdf,0x01,0xde,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80de01df <unknown>

// Multiple and single vectors

fmop4s  za0.d, {z0.d-z1.d}, z16.d  // 10000000-11000000-00000010-00011000
// CHECK-INST: fmop4s  za0.d, { z0.d, z1.d }, z16.d
// CHECK-ENCODING: [0x18,0x02,0xc0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c00218 <unknown>

fmop4s  za5.d, {z10.d-z11.d}, z20.d  // 10000000-11000100-00000011-01011101
// CHECK-INST: fmop4s  za5.d, { z10.d, z11.d }, z20.d
// CHECK-ENCODING: [0x5d,0x03,0xc4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80c4035d <unknown>

fmop4s  za7.d, {z14.d-z15.d}, z30.d  // 10000000-11001110-00000011-11011111
// CHECK-INST: fmop4s  za7.d, { z14.d, z15.d }, z30.d
// CHECK-ENCODING: [0xdf,0x03,0xce,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80ce03df <unknown>

// Multiple vectors

fmop4s  za0.d, {z0.d-z1.d}, {z16.d-z17.d}  // 10000000-11010000-00000010-00011000
// CHECK-INST: fmop4s  za0.d, { z0.d, z1.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x18,0x02,0xd0,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d00218 <unknown>

fmop4s  za5.d, {z10.d-z11.d}, {z20.d-z21.d}  // 10000000-11010100-00000011-01011101
// CHECK-INST: fmop4s  za5.d, { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x5d,0x03,0xd4,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80d4035d <unknown>

fmop4s  za7.d, {z14.d-z15.d}, {z30.d-z31.d}  // 10000000-11011110-00000011-11011111
// CHECK-INST: fmop4s  za7.d, { z14.d, z15.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xdf,0x03,0xde,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f64f64
// CHECK-UNKNOWN: 80de03df <unknown>
