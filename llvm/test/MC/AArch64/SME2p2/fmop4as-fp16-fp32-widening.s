
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
fmop4a  za0.s, z0.h, z16.h  // 10000001-00100000-00000000-00000000
// CHECK-INST: fmop4a  za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x00,0x00,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81200000 <unknown>

fmop4a  za1.s, z10.h, z20.h  // 10000001-00100100-00000001-01000001
// CHECK-INST: fmop4a  za1.s, z10.h, z20.h
// CHECK-ENCODING: [0x41,0x01,0x24,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81240141 <unknown>

fmop4a  za3.s, z14.h, z30.h  // 10000001-00101110-00000001-11000011
// CHECK-INST: fmop4a  za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xc3,0x01,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 812e01c3 <unknown>

// Single and multiple vectors

fmop4a  za0.s, z0.h, {z16.h-z17.h}  // 10000001-00110000-00000000-00000000
// CHECK-INST: fmop4a  za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x00,0x00,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81300000 <unknown>

fmop4a  za1.s, z10.h, {z20.h-z21.h}  // 10000001-00110100-00000001-01000001
// CHECK-INST: fmop4a  za1.s, z10.h, { z20.h, z21.h }
// CHECK-ENCODING: [0x41,0x01,0x34,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81340141 <unknown>

fmop4a  za3.s, z14.h, {z30.h-z31.h}  // 10000001-00111110-00000001-11000011
// CHECK-INST: fmop4a  za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xc3,0x01,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 813e01c3 <unknown>

// Multiple and single vectors

fmop4a  za0.s, {z0.h-z1.h}, z16.h  // 10000001-00100000-00000010-00000000
// CHECK-INST: fmop4a  za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x00,0x02,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81200200 <unknown>

fmop4a  za1.s, {z10.h-z11.h}, z20.h  // 10000001-00100100-00000011-01000001
// CHECK-INST: fmop4a  za1.s, { z10.h, z11.h }, z20.h
// CHECK-ENCODING: [0x41,0x03,0x24,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81240341 <unknown>

fmop4a  za3.s, {z14.h-z15.h}, z30.h  // 10000001-00101110-00000011-11000011
// CHECK-INST: fmop4a  za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xc3,0x03,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 812e03c3 <unknown>

// Multiple vectors

fmop4a  za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00110000-00000010-00000000
// CHECK-INST: fmop4a  za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x00,0x02,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81300200 <unknown>

fmop4a  za1.s, {z10.h-z11.h}, {z20.h-z21.h}  // 10000001-00110100-00000011-01000001
// CHECK-INST: fmop4a  za1.s, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x41,0x03,0x34,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81340341 <unknown>

fmop4a  za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00111110-00000011-11000011
// CHECK-INST: fmop4a  za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc3,0x03,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 813e03c3 <unknown>

// FMOP4S

// Single vectors
fmop4s  za0.s, z0.h, z16.h  // 10000001-00100000-00000000-00010000
// CHECK-INST: fmop4s  za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x10,0x00,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81200010 <unknown>

fmop4s  za1.s, z10.h, z20.h  // 10000001-00100100-00000001-01010001
// CHECK-INST: fmop4s  za1.s, z10.h, z20.h
// CHECK-ENCODING: [0x51,0x01,0x24,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81240151 <unknown>

fmop4s  za3.s, z14.h, z30.h  // 10000001-00101110-00000001-11010011
// CHECK-INST: fmop4s  za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xd3,0x01,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 812e01d3 <unknown>

// Single and multiple vectors

fmop4s  za0.s, z0.h, {z16.h-z17.h}  // 10000001-00110000-00000000-00010000
// CHECK-INST: fmop4s  za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x10,0x00,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81300010 <unknown>

fmop4s  za1.s, z10.h, {z20.h-z21.h}  // 10000001-00110100-00000001-01010001
// CHECK-INST: fmop4s  za1.s, z10.h, { z20.h, z21.h }
// CHECK-ENCODING: [0x51,0x01,0x34,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81340151 <unknown>

fmop4s  za3.s, z14.h, {z30.h-z31.h}  // 10000001-00111110-00000001-11010011
// CHECK-INST: fmop4s  za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x01,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 813e01d3 <unknown>

// Multiple and single vectors

fmop4s  za0.s, {z0.h-z1.h}, z16.h  // 10000001-00100000-00000010-00010000
// CHECK-INST: fmop4s  za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x10,0x02,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81200210 <unknown>

fmop4s  za1.s, {z10.h-z11.h}, z20.h  // 10000001-00100100-00000011-01010001
// CHECK-INST: fmop4s  za1.s, { z10.h, z11.h }, z20.h
// CHECK-ENCODING: [0x51,0x03,0x24,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81240351 <unknown>

fmop4s  za3.s, {z14.h-z15.h}, z30.h  // 10000001-00101110-00000011-11010011
// CHECK-INST: fmop4s  za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xd3,0x03,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 812e03d3 <unknown>

// Multiple vectors

fmop4s  za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00110000-00000010-00010000
// CHECK-INST: fmop4s  za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x10,0x02,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81300210 <unknown>

fmop4s  za1.s, {z10.h-z11.h}, {z20.h-z21.h}  // 10000001-00110100-00000011-01010001
// CHECK-INST: fmop4s  za1.s, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x51,0x03,0x34,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81340351 <unknown>

fmop4s  za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00111110-00000011-11010011
// CHECK-INST: fmop4s  za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x03,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 813e03d3 <unknown>
