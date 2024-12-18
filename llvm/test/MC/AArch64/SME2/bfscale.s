// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-bfscale < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sve-bfscale < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sve-bfscale - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sve-bfscale < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-bfscale < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sve-bfscale -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Multiple and single vector, 2 regs

bfscale {z0.h-z1.h}, {z0.h-z1.h}, z0.h  // 11000001-00100000-10100001-10000000
// CHECK-INST: bfscale { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x80,0xa1,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c120a180 <unknown>

bfscale {z20.h-z21.h}, {z20.h-z21.h}, z5.h  // 11000001-00100101-10100001-10010100
// CHECK-INST: bfscale { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x94,0xa1,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c125a194 <unknown>

bfscale {z30.h-z31.h}, {z30.h-z31.h}, z15.h  // 11000001-00101111-10100001-10011110
// CHECK-INST: bfscale { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x9e,0xa1,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c12fa19e <unknown>

// Multiple and single vector, 4 regs

bfscale {z0.h-z3.h}, {z0.h-z3.h}, z0.h  // 11000001-00100000-10101001-10000000
// CHECK-INST: bfscale { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x80,0xa9,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c120a980 <unknown>

bfscale {z20.h-z23.h}, {z20.h-z23.h}, z5.h  // 11000001-00100101-10101001-10010100
// CHECK-INST: bfscale { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x94,0xa9,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c125a994 <unknown>

bfscale {z28.h-z31.h}, {z28.h-z31.h}, z15.h  // 11000001-00101111-10101001-10011100
// CHECK-INST: bfscale { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x9c,0xa9,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c12fa99c <unknown>

// Multiple vectors, 2 regs

bfscale {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z1.h}  // 11000001-00100000-10110001-10000000
// CHECK-INST: bfscale { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x80,0xb1,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c120b180 <unknown>

bfscale {z20.h-z21.h}, {z20.h-z21.h}, {z20.h-z21.h}  // 11000001-00110100-10110001-10010100
// CHECK-INST: bfscale { z20.h, z21.h }, { z20.h, z21.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x94,0xb1,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c134b194 <unknown>

bfscale {z30.h-z31.h}, {z30.h-z31.h}, {z30.h-z31.h}  // 11000001-00111110-10110001-10011110
// CHECK-INST: bfscale { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x9e,0xb1,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c13eb19e <unknown>

// Multiple vectors, 4 regs

bfscale {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-00100000-10111001-10000000
// CHECK-INST: bfscale { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x80,0xb9,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c120b980 <unknown>

bfscale {z20.h-z23.h}, {z20.h-z23.h}, {z20.h-z23.h}  // 11000001-00110100-10111001-10010100
// CHECK-INST: bfscale { z20.h - z23.h }, { z20.h - z23.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x94,0xb9,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c134b994 <unknown>

bfscale {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-00111100-10111001-10011100
// CHECK-INST: bfscale { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9c,0xb9,0x3c,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c13cb99c <unknown>
