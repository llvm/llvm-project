// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 --mattr=+sme2p1,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfmax   {z0.h, z1.h}, {z0.h, z1.h}, z0.h  // 11000001-00100000-10100001-00000000
// CHECK-INST: bfmax   { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0xa1,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c120a100 <unknown>

bfmax   {z20.h, z21.h}, {z20.h, z21.h}, z5.h  // 11000001-00100101-10100001-00010100
// CHECK-INST: bfmax   { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x14,0xa1,0x25,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c125a114 <unknown>

bfmax   {z22.h, z23.h}, {z22.h, z23.h}, z8.h  // 11000001-00101000-10100001-00010110
// CHECK-INST: bfmax   { z22.h, z23.h }, { z22.h, z23.h }, z8.h
// CHECK-ENCODING: [0x16,0xa1,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c128a116 <unknown>

bfmax   {z30.h, z31.h}, {z30.h, z31.h}, z15.h  // 11000001-00101111-10100001-00011110
// CHECK-INST: bfmax   { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x1e,0xa1,0x2f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c12fa11e <unknown>

bfmax   {z0.h, z1.h}, {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-00100000-10110001-00000000
// CHECK-INST: bfmax   { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0xb1,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c120b100 <unknown>

bfmax   {z20.h, z21.h}, {z20.h, z21.h}, {z20.h, z21.h}  // 11000001-00110100-10110001-00010100
// CHECK-INST: bfmax   { z20.h, z21.h }, { z20.h, z21.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x14,0xb1,0x34,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c134b114 <unknown>

bfmax   {z22.h, z23.h}, {z22.h, z23.h}, {z8.h, z9.h}  // 11000001-00101000-10110001-00010110
// CHECK-INST: bfmax   { z22.h, z23.h }, { z22.h, z23.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x16,0xb1,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c128b116 <unknown>

bfmax   {z30.h, z31.h}, {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-00111110-10110001-00011110
// CHECK-INST: bfmax   { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x1e,0xb1,0x3e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c13eb11e <unknown>

bfmax   {z0.h - z3.h}, {z0.h - z3.h}, z0.h  // 11000001-00100000-10101001-00000000
// CHECK-INST: bfmax   { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0xa9,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c120a900 <unknown>

bfmax   {z20.h - z23.h}, {z20.h - z23.h}, z5.h  // 11000001-00100101-10101001-00010100
// CHECK-INST: bfmax   { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x14,0xa9,0x25,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c125a914 <unknown>

bfmax   {z20.h - z23.h}, {z20.h - z23.h}, z8.h  // 11000001-00101000-10101001-00010100
// CHECK-INST: bfmax   { z20.h - z23.h }, { z20.h - z23.h }, z8.h
// CHECK-ENCODING: [0x14,0xa9,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c128a914 <unknown>

bfmax   {z28.h - z31.h}, {z28.h - z31.h}, z15.h  // 11000001-00101111-10101001-00011100
// CHECK-INST: bfmax   { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x1c,0xa9,0x2f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c12fa91c <unknown>

bfmax   {z0.h - z3.h}, {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-00100000-10111001-00000000
// CHECK-INST: bfmax   { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0xb9,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c120b900 <unknown>

bfmax   {z20.h - z23.h}, {z20.h - z23.h}, {z20.h - z23.h}  // 11000001-00110100-10111001-00010100
// CHECK-INST: bfmax   { z20.h - z23.h }, { z20.h - z23.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x14,0xb9,0x34,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c134b914 <unknown>

bfmax   {z20.h - z23.h}, {z20.h - z23.h}, {z8.h - z11.h}  // 11000001-00101000-10111001-00010100
// CHECK-INST: bfmax   { z20.h - z23.h }, { z20.h - z23.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x14,0xb9,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c128b914 <unknown>

bfmax   {z28.h - z31.h}, {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-00111100-10111001-00011100
// CHECK-INST: bfmax   { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x1c,0xb9,0x3c,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c13cb91c <unknown>
