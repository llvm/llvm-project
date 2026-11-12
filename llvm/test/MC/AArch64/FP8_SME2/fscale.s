// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+fp8 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+fp8 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
//2X
fscale  {z0.h-z1.h}, {z0.h-z1.h}, z0.h  // 11000001-01100000-10100001-10000000
// CHECK-INST: fscale  { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x80,0xa1,0x60,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c160a180 <unknown>

fscale  {z30.h-z31.h}, {z30.h-z31.h}, z15.h  // 11000001-01101111-10100001-10011110
// CHECK-INST: fscale  { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x9e,0xa1,0x6f,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c16fa19e <unknown>

fscale  {z0.s-z1.s}, {z0.s-z1.s}, z0.s  // 11000001-10100000-10100001-10000000
// CHECK-INST: fscale  { z0.s, z1.s }, { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x80,0xa1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a0a180 <unknown>

fscale  {z30.s-z31.s}, {z30.s-z31.s}, z15.s  // 11000001-10101111-10100001-10011110
// CHECK-INST: fscale  { z30.s, z31.s }, { z30.s, z31.s }, z15.s
// CHECK-ENCODING: [0x9e,0xa1,0xaf,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1afa19e <unknown>

fscale  {z0.d-z1.d}, {z0.d-z1.d}, z0.d  // 11000001-11100000-10100001-10000000
// CHECK-INST: fscale  { z0.d, z1.d }, { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x80,0xa1,0xe0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e0a180 <unknown>

fscale  {z30.d-z31.d}, {z30.d-z31.d}, z15.d  // 11000001-11101111-10100001-10011110
// CHECK-INST: fscale  { z30.d, z31.d }, { z30.d, z31.d }, z15.d
// CHECK-ENCODING: [0x9e,0xa1,0xef,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1efa19e <unknown>

fscale  {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z1.h}  // 11000001-01100000-10110001-10000000
// CHECK-INST: fscale  { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x80,0xb1,0x60,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c160b180 <unknown>

fscale  {z30.h-z31.h}, {z30.h-z31.h}, {z30.h-z31.h}  // 11000001-01111110-10110001-10011110
// CHECK-INST: fscale  { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x9e,0xb1,0x7e,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c17eb19e <unknown>

fscale  {z0.s-z1.s}, {z0.s-z1.s}, {z0.s-z1.s}  // 11000001-10100000-10110001-10000000
// CHECK-INST: fscale  { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x80,0xb1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a0b180 <unknown>

fscale  {z30.s-z31.s}, {z30.s-z31.s}, {z30.s-z31.s}  // 11000001-10111110-10110001-10011110
// CHECK-INST: fscale  { z30.s, z31.s }, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x9e,0xb1,0xbe,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1beb19e <unknown>

fscale  {z0.d-z1.d}, {z0.d-z1.d}, {z0.d-z1.d}  // 11000001-11100000-10110001-10000000
// CHECK-INST: fscale  { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x80,0xb1,0xe0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e0b180 <unknown>

fscale  {z30.d-z31.d}, {z30.d-z31.d}, {z30.d-z31.d}  // 11000001-11111110-10110001-10011110
// CHECK-INST: fscale  { z30.d, z31.d }, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x9e,0xb1,0xfe,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1feb19e <unknown>


//4X

fscale  {z0.h-z3.h}, {z0.h-z3.h}, z0.h  // 11000001-01100000-10101001-10000000
// CHECK-INST: fscale  { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x80,0xa9,0x60,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c160a980 <unknown>

fscale  {z28.h-z31.h}, {z28.h-z31.h}, z15.h  // 11000001-01101111-10101001-10011100
// CHECK-INST: fscale  { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x9c,0xa9,0x6f,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c16fa99c <unknown>

fscale  {z0.s-z3.s}, {z0.s-z3.s}, z0.s  // 11000001-10100000-10101001-10000000
// CHECK-INST: fscale  { z0.s - z3.s }, { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x80,0xa9,0xa0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a0a980 <unknown>

fscale  {z28.s-z31.s}, {z28.s-z31.s}, z15.s  // 11000001-10101111-10101001-10011100
// CHECK-INST: fscale  { z28.s - z31.s }, { z28.s - z31.s }, z15.s
// CHECK-ENCODING: [0x9c,0xa9,0xaf,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1afa99c <unknown>

fscale  {z0.d-z3.d}, {z0.d-z3.d}, z0.d  // 11000001-11100000-10101001-10000000
// CHECK-INST: fscale  { z0.d - z3.d }, { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x80,0xa9,0xe0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e0a980 <unknown>

fscale  {z28.d-z31.d}, {z28.d-z31.d}, z15.d  // 11000001-11101111-10101001-10011100
// CHECK-INST: fscale  { z28.d - z31.d }, { z28.d - z31.d }, z15.d
// CHECK-ENCODING: [0x9c,0xa9,0xef,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1efa99c <unknown>

fscale  {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-01100000-10111001-10000000
// CHECK-INST: fscale  { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x80,0xb9,0x60,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c160b980 <unknown>

fscale  {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-01111100-10111001-10011100
// CHECK-INST: fscale  { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9c,0xb9,0x7c,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c17cb99c <unknown>

fscale  {z0.s-z3.s}, {z0.s-z3.s}, {z0.s-z3.s}  // 11000001-10100000-10111001-10000000
// CHECK-INST: fscale  { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x80,0xb9,0xa0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a0b980 <unknown>

fscale  {z28.s-z31.s}, {z28.s-z31.s}, {z28.s-z31.s}  // 11000001-10111100-10111001-10011100
// CHECK-INST: fscale  { z28.s - z31.s }, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0xb9,0xbc,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1bcb99c <unknown>

fscale  {z0.d-z3.d}, {z0.d-z3.d}, {z0.d-z3.d}  // 11000001-11100000-10111001-10000000
// CHECK-INST: fscale  { z0.d - z3.d }, { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x80,0xb9,0xe0,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e0b980 <unknown>

fscale  {z28.d-z31.d}, {z28.d-z31.d}, {z28.d-z31.d}  // 11000001-11111100-10111001-10011100
// CHECK-INST: fscale  { z28.d - z31.d }, { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9c,0xb9,0xfc,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1fcb99c <unknown>
