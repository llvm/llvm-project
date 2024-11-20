// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-f16f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d  --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmopa   za0.h, p0/m, p0/m, z0.h, z0.h  // 10000001-10000000-00000000-00001000
// CHECK-INST: fmopa   za0.h, p0/m, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x08,0x00,0x80,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 81800008 <unknown>

fmopa   za1.h, p5/m, p2/m, z10.h, z21.h  // 10000001-10010101-01010101-01001001
// CHECK-INST: fmopa   za1.h, p5/m, p2/m, z10.h, z21.h
// CHECK-ENCODING: [0x49,0x55,0x95,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 81955549 <unknown>

fmopa   za1.h, p3/m, p7/m, z13.h, z8.h  // 10000001-10001000-11101101-10101001
// CHECK-INST: fmopa   za1.h, p3/m, p7/m, z13.h, z8.h
// CHECK-ENCODING: [0xa9,0xed,0x88,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 8188eda9 <unknown>

fmopa   za1.h, p7/m, p7/m, z31.h, z31.h  // 10000001-10011111-11111111-11101001
// CHECK-INST: fmopa   za1.h, p7/m, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xe9,0xff,0x9f,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 819fffe9 <unknown>

fmopa   za1.h, p3/m, p0/m, z17.h, z16.h  // 10000001-10010000-00001110-00101001
// CHECK-INST: fmopa   za1.h, p3/m, p0/m, z17.h, z16.h
// CHECK-ENCODING: [0x29,0x0e,0x90,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 81900e29 <unknown>

fmopa   za1.h, p1/m, p4/m, z1.h, z30.h  // 10000001-10011110-10000100-00101001
// CHECK-INST: fmopa   za1.h, p1/m, p4/m, z1.h, z30.h
// CHECK-ENCODING: [0x29,0x84,0x9e,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 819e8429 <unknown>

fmopa   za0.h, p5/m, p2/m, z19.h, z20.h  // 10000001-10010100-01010110-01101000
// CHECK-INST: fmopa   za0.h, p5/m, p2/m, z19.h, z20.h
// CHECK-ENCODING: [0x68,0x56,0x94,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 81945668 <unknown>

fmopa   za0.h, p6/m, p0/m, z12.h, z2.h  // 10000001-10000010-00011001-10001000
// CHECK-INST: fmopa   za0.h, p6/m, p0/m, z12.h, z2.h
// CHECK-ENCODING: [0x88,0x19,0x82,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 81821988 <unknown>

fmopa   za1.h, p2/m, p6/m, z1.h, z26.h  // 10000001-10011010-11001000-00101001
// CHECK-INST: fmopa   za1.h, p2/m, p6/m, z1.h, z26.h
// CHECK-ENCODING: [0x29,0xc8,0x9a,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 819ac829 <unknown>

fmopa   za1.h, p2/m, p0/m, z22.h, z30.h  // 10000001-10011110-00001010-11001001
// CHECK-INST: fmopa   za1.h, p2/m, p0/m, z22.h, z30.h
// CHECK-ENCODING: [0xc9,0x0a,0x9e,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 819e0ac9 <unknown>

fmopa   za0.h, p5/m, p7/m, z9.h, z1.h  // 10000001-10000001-11110101-00101000
// CHECK-INST: fmopa   za0.h, p5/m, p7/m, z9.h, z1.h
// CHECK-ENCODING: [0x28,0xf5,0x81,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 8181f528 <unknown>

fmopa   za1.h, p2/m, p5/m, z12.h, z11.h  // 10000001-10001011-10101001-10001001
// CHECK-INST: fmopa   za1.h, p2/m, p5/m, z12.h, z11.h
// CHECK-ENCODING: [0x89,0xa9,0x8b,0x81]
// CHECK-ERROR: instruction requires: sme-f16f16
// CHECK-UNKNOWN: 818ba989 <unknown>

