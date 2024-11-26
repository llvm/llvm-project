
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


// FMOP4A

// Single vectors

fmop4a  za0.s, z0.s, z16.s  // 10000000-00000000-00000000-00000000
// CHECK-INST: fmop4a  za0.s, z0.s, z16.s
// CHECK-ENCODING: [0x00,0x00,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80000000 <unknown>

fmop4a  za3.s, z12.s, z24.s  // 10000000-00001000-00000001-10000011
// CHECK-INST: fmop4a  za3.s, z12.s, z24.s
// CHECK-ENCODING: [0x83,0x01,0x08,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80080183 <unknown>

fmop4a  za3.s, z14.s, z30.s  // 10000000-00001110-00000001-11000011
// CHECK-INST: fmop4a  za3.s, z14.s, z30.s
// CHECK-ENCODING: [0xc3,0x01,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e01c3 <unknown>

// Single and multiple vectors

fmop4a  za0.s, z0.s, {z16.s-z17.s}  // 10000000-00010000-00000000-00000000
// CHECK-INST: fmop4a  za0.s, z0.s, { z16.s, z17.s }
// CHECK-ENCODING: [0x00,0x00,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80100000 <unknown>

fmop4a  za1.s, z10.s, {z20.s-z21.s}  // 10000000-00010100-00000001-01000001
// CHECK-INST: fmop4a  za1.s, z10.s, { z20.s, z21.s }
// CHECK-ENCODING: [0x41,0x01,0x14,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80140141 <unknown>

fmop4a  za3.s, z14.s, {z30.s-z31.s}  // 10000000-00011110-00000001-11000011
// CHECK-INST: fmop4a  za3.s, z14.s, { z30.s, z31.s }
// CHECK-ENCODING: [0xc3,0x01,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e01c3 <unknown>

// Multiple and single vectors

fmop4a  za0.s, {z0.s-z1.s}, z16.s  // 10000000-00000000-00000010-00000000
// CHECK-INST: fmop4a  za0.s, { z0.s, z1.s }, z16.s
// CHECK-ENCODING: [0x00,0x02,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80000200 <unknown>

fmop4a  za1.s, {z10.s-z11.s}, z20.s  // 10000000-00000100-00000011-01000001
// CHECK-INST: fmop4a  za1.s, { z10.s, z11.s }, z20.s
// CHECK-ENCODING: [0x41,0x03,0x04,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80040341 <unknown>

fmop4a  za3.s, {z14.s-z15.s}, z30.s  // 10000000-00001110-00000011-11000011
// CHECK-INST: fmop4a  za3.s, { z14.s, z15.s }, z30.s
// CHECK-ENCODING: [0xc3,0x03,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e03c3 <unknown>

// Multiple vectors

fmop4a  za0.s, {z0.s-z1.s}, {z16.s-z17.s}  // 10000000-00010000-00000010-00000000
// CHECK-INST: fmop4a  za0.s, { z0.s, z1.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x00,0x02,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80100200 <unknown>

fmop4a  za1.s, {z10.s-z11.s}, {z20.s-z21.s}  // 10000000-00010100-00000011-01000001
// CHECK-INST: fmop4a  za1.s, { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x41,0x03,0x14,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80140341 <unknown>

fmop4a  za3.s, {z14.s-z15.s}, {z30.s-z31.s}  // 10000000-00011110-00000011-11000011
// CHECK-INST: fmop4a  za3.s, { z14.s, z15.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xc3,0x03,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e03c3 <unknown>

// FMOP4S

// Single vectors

fmop4s  za0.s, z0.s, z16.s  // 10000000-00000000-00000000-00010000
// CHECK-INST: fmop4s  za0.s, z0.s, z16.s
// CHECK-ENCODING: [0x10,0x00,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80000010 <unknown>

fmop4s  za3.s, z12.s, z24.s  // 10000000-00001000-00000001-10010011
// CHECK-INST: fmop4s  za3.s, z12.s, z24.s
// CHECK-ENCODING: [0x93,0x01,0x08,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80080193 <unknown>

fmop4s  za3.s, z14.s, z30.s  // 10000000-00001110-00000001-11010011
// CHECK-INST: fmop4s  za3.s, z14.s, z30.s
// CHECK-ENCODING: [0xd3,0x01,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e01d3 <unknown>

// Single and multiple vectors

fmop4s  za0.s, z0.s, {z16.s-z17.s}  // 10000000-00010000-00000000-00010000
// CHECK-INST: fmop4s  za0.s, z0.s, { z16.s, z17.s }
// CHECK-ENCODING: [0x10,0x00,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80100010 <unknown>

fmop4s  za1.s, z10.s, {z20.s-z21.s}  // 10000000-00010100-00000001-01010001
// CHECK-INST: fmop4s  za1.s, z10.s, { z20.s, z21.s }
// CHECK-ENCODING: [0x51,0x01,0x14,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80140151 <unknown>

fmop4s  za3.s, z14.s, {z30.s-z31.s}  // 10000000-00011110-00000001-11010011
// CHECK-INST: fmop4s  za3.s, z14.s, { z30.s, z31.s }
// CHECK-ENCODING: [0xd3,0x01,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e01d3 <unknown>

// Multiple and single vectors

fmop4s  za0.s, {z0.s-z1.s}, z16.s  // 10000000-00000000-00000010-00010000
// CHECK-INST: fmop4s  za0.s, { z0.s, z1.s }, z16.s
// CHECK-ENCODING: [0x10,0x02,0x00,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80000210 <unknown>

fmop4s  za1.s, {z10.s-z11.s}, z20.s  // 10000000-00000100-00000011-01010001
// CHECK-INST: fmop4s  za1.s, { z10.s, z11.s }, z20.s
// CHECK-ENCODING: [0x51,0x03,0x04,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80040351 <unknown>

fmop4s  za3.s, {z14.s-z15.s}, z30.s  // 10000000-00001110-00000011-11010011
// CHECK-INST: fmop4s  za3.s, { z14.s, z15.s }, z30.s
// CHECK-ENCODING: [0xd3,0x03,0x0e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 800e03d3 <unknown>

// Multiple vectors

fmop4s  za0.s, {z0.s-z1.s}, {z16.s-z17.s}  // 10000000-00010000-00000010-00010000
// CHECK-INST: fmop4s  za0.s, { z0.s, z1.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x10,0x02,0x10,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80100210 <unknown>

fmop4s  za1.s, {z10.s-z11.s}, {z20.s-z21.s}  // 10000000-00010100-00000011-01010001
// CHECK-INST: fmop4s  za1.s, { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x51,0x03,0x14,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80140351 <unknown>

fmop4s  za3.s, {z14.s-z15.s}, {z30.s-z31.s}  // 10000000-00011110-00000011-11010011
// CHECK-INST: fmop4s  za3.s, { z14.s, z15.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xd3,0x03,0x1e,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 801e03d3 <unknown>
