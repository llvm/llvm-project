// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+faminmax < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+faminmax < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+faminmax - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+faminmax < %s \
// RUN:        | llvm-objdump -d --mattr=-faminmax - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+faminmax < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+faminmax -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// FAMAX
famax   {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z1.h}  // 11000001-01100000-10110001-01000000
// CHECK-INST: famax   { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x40,0xb1,0x60,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c160b140 <unknown>

famax   {z30.h-z31.h}, {z30.h-z31.h}, {z30.h-z31.h}  // 11000001-01111110-10110001-01011110
// CHECK-INST: famax   { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x5e,0xb1,0x7e,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c17eb15e <unknown>

famax   {z0.s-z1.s}, {z0.s-z1.s}, {z0.s-z1.s}  // 11000001-10100000-10110001-01000000
// CHECK-INST: famax   { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x40,0xb1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1a0b140 <unknown>

famax   {z30.s-z31.s}, {z30.s-z31.s}, {z30.s-z31.s}  // 11000001-10111110-10110001-01011110
// CHECK-INST: famax   { z30.s, z31.s }, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x5e,0xb1,0xbe,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1beb15e <unknown>

famax   {z0.d-z1.d}, {z0.d-z1.d}, {z0.d-z1.d}  // 11000001-11100000-10110001-01000000
// CHECK-INST: famax   { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x40,0xb1,0xe0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1e0b140 <unknown>

famax   {z30.d-z31.d}, {z30.d-z31.d}, {z30.d-z31.d}  // 11000001-11111110-10110001-01011110
// CHECK-INST: famax   { z30.d, z31.d }, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x5e,0xb1,0xfe,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1feb15e <unknown>

famax   {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-01100000-10111001-01000000
// CHECK-INST: famax   { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x40,0xb9,0x60,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c160b940 <unknown>

famax   {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-01111100-10111001-01011100
// CHECK-INST: famax   { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x5c,0xb9,0x7c,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c17cb95c <unknown>

famax   {z0.s-z3.s}, {z0.s-z3.s}, {z0.s-z3.s}  // 11000001-10100000-10111001-01000000
// CHECK-INST: famax   { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x40,0xb9,0xa0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1a0b940 <unknown>

famax   {z28.s-z31.s}, {z28.s-z31.s}, {z28.s-z31.s}  // 11000001-10111100-10111001-01011100
// CHECK-INST: famax   { z28.s - z31.s }, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x5c,0xb9,0xbc,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1bcb95c <unknown>

famax   {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-01100000-10111001-01000000
// CHECK-INST: famax   { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x40,0xb9,0x60,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c160b940 <unknown>

famax   {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-01111100-10111001-01011100
// CHECK-INST: famax   { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x5c,0xb9,0x7c,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c17cb95c <unknown>

// FAMIN
famin   {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z1.h}  // 11000001-01100000-10110001-01000001
// CHECK-INST: famin   { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x41,0xb1,0x60,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c160b141 <unknown>

famin   {z30.h-z31.h}, {z30.h-z31.h}, {z30.h-z31.h}  // 11000001-01111110-10110001-01011111
// CHECK-INST: famin   { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x5f,0xb1,0x7e,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c17eb15f <unknown>

famin   {z0.s-z1.s}, {z0.s-z1.s}, {z0.s-z1.s}  // 11000001-10100000-10110001-01000001
// CHECK-INST: famin   { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x41,0xb1,0xa0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1a0b141 <unknown>

famin   {z30.s-z31.s}, {z30.s-z31.s}, {z30.s-z31.s}  // 11000001-10111110-10110001-01011111
// CHECK-INST: famin   { z30.s, z31.s }, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x5f,0xb1,0xbe,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1beb15f <unknown>

famin   {z0.d-z1.d}, {z0.d-z1.d}, {z0.d-z1.d}  // 11000001-11100000-10110001-01000001
// CHECK-INST: famin   { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x41,0xb1,0xe0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1e0b141 <unknown>

famin   {z30.d-z31.d}, {z30.d-z31.d}, {z30.d-z31.d}  // 11000001-11111110-10110001-01011111
// CHECK-INST: famin   { z30.d, z31.d }, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x5f,0xb1,0xfe,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1feb15f <unknown>

famin   {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-01100000-10111001-01000001
// CHECK-INST: famin   { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x41,0xb9,0x60,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c160b941 <unknown>

famin   {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-01111100-10111001-01011101
// CHECK-INST: famin   { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x5d,0xb9,0x7c,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c17cb95d <unknown>

famin   {z0.s-z3.s}, {z0.s-z3.s}, {z0.s-z3.s}  // 11000001-10100000-10111001-01000001
// CHECK-INST: famin   { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x41,0xb9,0xa0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1a0b941 <unknown>

famin   {z28.s-z31.s}, {z28.s-z31.s}, {z28.s-z31.s}  // 11000001-10111100-10111001-01011101
// CHECK-INST: famin   { z28.s - z31.s }, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x5d,0xb9,0xbc,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1bcb95d <unknown>

famin   {z0.d-z3.d}, {z0.d-z3.d}, {z0.d-z3.d}  // 11000001-11100000-10111001-01000001
// CHECK-INST: famin   { z0.d - z3.d }, { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x41,0xb9,0xe0,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1e0b941 <unknown>

famin   {z28.d-z31.d}, {z28.d-z31.d}, {z28.d-z31.d}  // 11000001-11111100-10111001-01011101
// CHECK-INST: famin   { z28.d - z31.d }, { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x5d,0xb9,0xfc,0xc1]
// CHECK-ERROR: instruction requires: faminmax sme2
// CHECK-UNKNOWN: c1fcb95d <unknown>
