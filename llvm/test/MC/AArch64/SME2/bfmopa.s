// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfmopa  za0.h, p0/m, p0/m, z0.h, z0.h  // 10000001-10100000-00000000-00001000
// CHECK-INST: bfmopa  za0.h, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x08,0x00,0xa0,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81a00008 <unknown>

bfmopa  za1.h, p5/m, p2/m, z10.h, z21.h  // 10000001-10110101-01010101-01001001
// CHECK-INST: bfmopa  za1.h, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x49,0x55,0xb5,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81b55549 <unknown>

bfmopa  za1.h, p3/m, p7/m, z13.h, z8.h  // 10000001-10101000-11101101-10101001
// CHECK-INST: bfmopa  za1.h, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xa9,0xed,0xa8,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81a8eda9 <unknown>

bfmopa  za1.h, p7/m, p7/m, z31.h, z31.h  // 10000001-10111111-11111111-11101001
// CHECK-INST: bfmopa  za1.h, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xe9,0xff,0xbf,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81bfffe9 <unknown>

bfmopa  za1.h, p3/m, p0/m, z17.h, z16.h  // 10000001-10110000-00001110-00101001
// CHECK-INST: bfmopa  za1.h, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x29,0x0e,0xb0,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81b00e29 <unknown>

bfmopa  za1.h, p1/m, p4/m, z1.h, z30.h  // 10000001-10111110-10000100-00101001
// CHECK-INST: bfmopa  za1.h, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x29,0x84,0xbe,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81be8429 <unknown>

bfmopa  za0.h, p5/m, p2/m, z19.h, z20.h  // 10000001-10110100-01010110-01101000
// CHECK-INST: bfmopa  za0.h, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x68,0x56,0xb4,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81b45668 <unknown>

bfmopa  za0.h, p6/m, p0/m, z12.h, z2.h  // 10000001-10100010-00011001-10001000
// CHECK-INST: bfmopa  za0.h, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x88,0x19,0xa2,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81a21988 <unknown>

bfmopa  za1.h, p2/m, p6/m, z1.h, z26.h  // 10000001-10111010-11001000-00101001
// CHECK-INST: bfmopa  za1.h, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x29,0xc8,0xba,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81bac829 <unknown>

bfmopa  za1.h, p2/m, p0/m, z22.h, z30.h  // 10000001-10111110-00001010-11001001
// CHECK-INST: bfmopa  za1.h, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xc9,0x0a,0xbe,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81be0ac9 <unknown>

bfmopa  za0.h, p5/m, p7/m, z9.h, z1.h  // 10000001-10100001-11110101-00101000
// CHECK-INST: bfmopa  za0.h, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x28,0xf5,0xa1,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81a1f528 <unknown>

bfmopa  za1.h, p2/m, p5/m, z12.h, z11.h  // 10000001-10101011-10101001-10001001
// CHECK-INST: bfmopa  za1.h, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x89,0xa9,0xab,0x81]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: 81aba989 <unknown>
