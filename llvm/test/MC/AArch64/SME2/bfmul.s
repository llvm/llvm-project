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

// Multiple and single, 2 regs

bfmul   {z0.h-z1.h}, {z0.h-z1.h}, z0.h  // 11000001-00100000-11101000-00000000
// CHECK-INST: bfmul   { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0xe8,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c120e800 <unknown>

bfmul   {z20.h-z21.h}, {z10.h-z11.h}, z10.h  // 11000001-00110100-11101001-01010100
// CHECK-INST: bfmul   { z20.h, z21.h }, { z10.h, z11.h }, z10.h
// CHECK-ENCODING: [0x54,0xe9,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c134e954 <unknown>

bfmul   {z30.h-z31.h}, {z30.h-z31.h}, z15.h  // 11000001-00111110-11101011-11011110
// CHECK-INST: bfmul   { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0xde,0xeb,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c13eebde <unknown>

// Multiple and single, 4 regs

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, z0.h  // 11000001-00100001-11101000-00000000
// CHECK-INST: bfmul   { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0xe8,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c121e800 <unknown>

bfmul   {z20.h-z23.h}, {z8.h-z11.h}, z10.h  // 11000001-00110101-11101001-00010100
// CHECK-INST: bfmul   { z20.h - z23.h }, { z8.h - z11.h }, z10.h
// CHECK-ENCODING: [0x14,0xe9,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c135e914 <unknown>

bfmul   {z28.h-z31.h}, {z28.h-z31.h}, z15.h  // 11000001-00111111-11101011-10011100
// CHECK-INST: bfmul   { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x9c,0xeb,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c13feb9c <unknown>

// Multiple, 2 regs
bfmul   {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z1.h}  // 11000001-00100000-11100100-00000000
// CHECK-INST: bfmul   { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0xe4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c120e400 <unknown>

bfmul   {z20.h-z21.h}, {z10.h-z11.h}, {z20.h-z21.h}  // 11000001-00110100-11100101-01010100
// CHECK-INST: bfmul   { z20.h, z21.h }, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x54,0xe5,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c134e554 <unknown>

bfmul   {z30.h-z31.h}, {z30.h-z31.h}, {z30.h-z31.h}  // 11000001-00111110-11100111-11011110
// CHECK-INST: bfmul   { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xde,0xe7,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c13ee7de <unknown>

// Multiple, 4 regs

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-00100001-11100100-00000000
// CHECK-INST: bfmul   { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0xe4,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c121e400 <unknown>

bfmul   {z20.h-z23.h}, {z8.h-z11.h}, {z20.h-z23.h}  // 11000001-00110101-11100101-00010100
// CHECK-INST: bfmul   { z20.h - z23.h }, { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x14,0xe5,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c135e514 <unknown>

bfmul   {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-00111101-11100111-10011100
// CHECK-INST: bfmul   { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9c,0xe7,0x3d,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-bfscale
// CHECK-UNKNOWN: c13de79c <unknown>
