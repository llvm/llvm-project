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

// BFMOP4A

// Single vectors

bfmop4a za0.s, z0.h, z16.h  // 10000001-00000000-00000000-00000000
// CHECK-INST: bfmop4a za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x00,0x00,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81000000 <unknown>

bfmop4a za3.s, z14.h, z30.h  // 10000001-00001110-00000001-11000011
// CHECK-INST: bfmop4a za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xc3,0x01,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e01c3 <unknown>

bfmop4a za1.s, z10.h, z20.h  // 10000001-00000100-00000001-01000001
// CHECK-INST: bfmop4a za1.s, z10.h, z20.h
// CHECK-ENCODING: [0x41,0x01,0x04,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81040141 <unknown>

// Single and multiple vectors

bfmop4a za0.s, z0.h, {z16.h-z17.h}  // 10000001-00010000-00000000-00000000
// CHECK-INST: bfmop4a za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x00,0x00,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81100000 <unknown>

bfmop4a za3.s, z14.h, {z30.h-z31.h}  // 10000001-00011110-00000001-11000011
// CHECK-INST: bfmop4a za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xc3,0x01,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e01c3 <unknown>

bfmop4a za2.s, z12.h, {z24.h-z25.h}  // 10000001-00011000-00000001-10000010
// CHECK-INST: bfmop4a za2.s, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x82,0x01,0x18,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81180182 <unknown>

// Multiple and single vectors

bfmop4a za0.s, {z0.h-z1.h}, z16.h  // 10000001-00000000-00000010-00000000
// CHECK-INST: bfmop4a za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x00,0x02,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81000200 <unknown>

bfmop4a za3.s, {z14.h-z15.h}, z30.h  // 10000001-00001110-00000011-11000011
// CHECK-INST: bfmop4a za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xc3,0x03,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e03c3 <unknown>

bfmop4a za2.s, {z12.h-z13.h}, z28.h  // 10000001-00001100-00000011-10000010
// CHECK-INST: bfmop4a za2.s, { z12.h, z13.h }, z28.h
// CHECK-ENCODING: [0x82,0x03,0x0c,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810c0382 <unknown>

// Multiple vectors

bfmop4a za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00010000-00000010-00000000
// CHECK-INST: bfmop4a za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x00,0x02,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81100200 <unknown>

bfmop4a za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00011110-00000011-11000011
// CHECK-INST: bfmop4a za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc3,0x03,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e03c3 <unknown>

bfmop4a za2.s, {z12.h-z13.h}, {z26.h-z27.h}  // 10000001-00011010-00000011-10000010
// CHECK-INST: bfmop4a za2.s, { z12.h, z13.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x82,0x03,0x1a,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811a0382 <unknown>


// BFMOP4S

// Single vectors

bfmop4s za0.s, z0.h, z16.h  // 10000001-00000000-00000000-00010000
// CHECK-INST: bfmop4s za0.s, z0.h, z16.h
// CHECK-ENCODING: [0x10,0x00,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81000010 <unknown>

bfmop4s za3.s, z14.h, z30.h  // 10000001-00001110-00000001-11010011
// CHECK-INST: bfmop4s za3.s, z14.h, z30.h
// CHECK-ENCODING: [0xd3,0x01,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e01d3 <unknown>

bfmop4s za1.s, z10.h, z20.h  // 10000001-00000100-00000001-01010001
// CHECK-INST: bfmop4s za1.s, z10.h, z20.h
// CHECK-ENCODING: [0x51,0x01,0x04,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81040151 <unknown>

// Single and multiple vectors

bfmop4s za0.s, z0.h, {z16.h-z17.h}  // 10000001-00010000-00000000-00010000
// CHECK-INST: bfmop4s za0.s, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x10,0x00,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81100010 <unknown>

bfmop4s za3.s, z14.h, {z30.h-z31.h}  // 10000001-00011110-00000001-11010011
// CHECK-INST: bfmop4s za3.s, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x01,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e01d3 <unknown>

bfmop4s za2.s, z12.h, {z24.h-z25.h}  // 10000001-00011000-00000001-10010010
// CHECK-INST: bfmop4s za2.s, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x92,0x01,0x18,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81180192 <unknown>

// Multiple and single vectors

bfmop4s za0.s, {z0.h-z1.h}, z16.h  // 10000001-00000000-00000010-00010000
// CHECK-INST: bfmop4s za0.s, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x10,0x02,0x00,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81000210 <unknown>

bfmop4s za3.s, {z14.h-z15.h}, z30.h  // 10000001-00001110-00000011-11010011
// CHECK-INST: bfmop4s za3.s, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xd3,0x03,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810e03d3 <unknown>

bfmop4s za2.s, {z12.h-z13.h}, z28.h  // 10000001-00001100-00000011-10010010
// CHECK-INST: bfmop4s za2.s, { z12.h, z13.h }, z28.h
// CHECK-ENCODING: [0x92,0x03,0x0c,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 810c0392 <unknown>

// Multiple vectors

bfmop4s za0.s, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00010000-00000010-00010000
// CHECK-INST: bfmop4s za0.s, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x10,0x02,0x10,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 81100210 <unknown>

bfmop4s za3.s, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00011110-00000011-11010011
// CHECK-INST: bfmop4s za3.s, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd3,0x03,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811e03d3 <unknown>

bfmop4s za2.s, {z12.h-z13.h}, {z26.h-z27.h}  // 10000001-00011010-00000011-10010010
// CHECK-INST: bfmop4s za2.s, { z12.h, z13.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x92,0x03,0x1a,0x81]
// CHECK-ERROR: instruction requires: sme-mop4
// CHECK-UNKNOWN: 811a0392 <unknown>
