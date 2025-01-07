// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// BFMOP4A

// Single vectors

bfmop4a za0.h, z0.h, z16.h  // 10000001-00100000-00000000-00001000
// CHECK-INST: bfmop4a za0.h, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x00,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81200008 <unknown>

bfmop4a za1.h, z12.h, z24.h  // 10000001-00101000-00000001-10001001
// CHECK-INST: bfmop4a za1.h, z12.h, z24.h
// CHECK-ENCODING: [0x89,0x01,0x28,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81280189 <unknown>

bfmop4a za1.h, z14.h, z30.h  // 10000001-00101110-00000001-11001001
// CHECK-INST: bfmop4a za1.h, z14.h, z30.h
// CHECK-ENCODING: [0xc9,0x01,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 812e01c9 <unknown>

// Single and multiple vectors

bfmop4a za0.h, z0.h, {z16.h-z17.h}  // 10000001-00110000-00000000-00001000
// CHECK-INST: bfmop4a za0.h, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x00,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81300008 <unknown>

bfmop4a za1.h, z12.h, {z24.h-z25.h}  // 10000001-00111000-00000001-10001001
// CHECK-INST: bfmop4a za1.h, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x89,0x01,0x38,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81380189 <unknown>

bfmop4a za1.h, z14.h, {z30.h-z31.h}  // 10000001-00111110-00000001-11001001
// CHECK-INST: bfmop4a za1.h, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xc9,0x01,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 813e01c9 <unknown>

// Multiple and single vectors

bfmop4a za0.h, {z0.h-z1.h}, z16.h  // 10000001-00100000-00000010-00001000
// CHECK-INST: bfmop4a za0.h, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x02,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81200208 <unknown>

bfmop4a za1.h, {z12.h-z13.h}, z24.h  // 10000001-00101000-00000011-10001001
// CHECK-INST: bfmop4a za1.h, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x89,0x03,0x28,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81280389 <unknown>

bfmop4a za1.h, {z14.h-z15.h}, z30.h  // 10000001-00101110-00000011-11001001
// CHECK-INST: bfmop4a za1.h, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xc9,0x03,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 812e03c9 <unknown>

// Multiple vectors

bfmop4a za0.h, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00110000-00000010-00001000
// CHECK-INST: bfmop4a za0.h, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x02,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81300208 <unknown>

bfmop4a za1.h, {z12.h-z13.h}, {z24.h-z25.h}  // 10000001-00111000-00000011-10001001
// CHECK-INST: bfmop4a za1.h, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x89,0x03,0x38,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81380389 <unknown>

bfmop4a za1.h, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00111110-00000011-11001001
// CHECK-INST: bfmop4a za1.h, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc9,0x03,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 813e03c9 <unknown>


// BFMOP4S

// Single vectors

bfmop4s za0.h, z0.h, z16.h  // 10000001-00100000-00000000-00011000
// CHECK-INST: bfmop4s za0.h, z0.h, z16.h
// CHECK-ENCODING: [0x18,0x00,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81200018 <unknown>

bfmop4s za1.h, z12.h, z24.h  // 10000001-00101000-00000001-10011001
// CHECK-INST: bfmop4s za1.h, z12.h, z24.h
// CHECK-ENCODING: [0x99,0x01,0x28,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81280199 <unknown>

bfmop4s za1.h, z14.h, z30.h  // 10000001-00101110-00000001-11011001
// CHECK-INST: bfmop4s za1.h, z14.h, z30.h
// CHECK-ENCODING: [0xd9,0x01,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 812e01d9 <unknown>

// Single and multiple vectors

bfmop4s za0.h, z0.h, {z16.h-z17.h}  // 10000001-00110000-00000000-00011000
// CHECK-INST: bfmop4s za0.h, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x00,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81300018 <unknown>

bfmop4s za1.h, z12.h, {z24.h-z25.h}  // 10000001-00111000-00000001-10011001
// CHECK-INST: bfmop4s za1.h, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x99,0x01,0x38,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81380199 <unknown>

bfmop4s za1.h, z14.h, {z30.h-z31.h}  // 10000001-00111110-00000001-11011001
// CHECK-INST: bfmop4s za1.h, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x01,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 813e01d9 <unknown>

// Multiple and single vectors

bfmop4s za0.h, {z0.h-z1.h}, z16.h  // 10000001-00100000-00000010-00011000
// CHECK-INST: bfmop4s za0.h, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x18,0x02,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81200218 <unknown>

bfmop4s za1.h, {z12.h-z13.h}, z24.h  // 10000001-00101000-00000011-10011001
// CHECK-INST: bfmop4s za1.h, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x99,0x03,0x28,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81280399 <unknown>

bfmop4s za1.h, {z14.h-z15.h}, z30.h  // 10000001-00101110-00000011-11011001
// CHECK-INST: bfmop4s za1.h, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xd9,0x03,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 812e03d9 <unknown>

// Multiple vectors

bfmop4s za0.h, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00110000-00000010-00011000
// CHECK-INST: bfmop4s za0.h, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x02,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81300218 <unknown>

bfmop4s za1.h, {z12.h-z13.h}, {z24.h-z25.h}  // 10000001-00111000-00000011-10011001
// CHECK-INST: bfmop4s za1.h, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x99,0x03,0x38,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81380399 <unknown>

bfmop4s za1.h, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00111110-00000011-11011001
// CHECK-INST: bfmop4s za1.h, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x03,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 813e03d9 <unknown>
