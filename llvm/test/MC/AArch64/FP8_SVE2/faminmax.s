// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+faminmax < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+faminmax < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+faminmax < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+faminmax - | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+faminmax < %s \
// RUN:        | llvm-objdump -d --mattr=-faminmax - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+faminmax < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+faminmax -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// FAMIN

famin   z0.h, p0/m, z0.h, z1.h  // 01100101-01001111-10000000-00100000
// CHECK-INST: famin   z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0x80,0x4f,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 654f8020 <unknown>

movprfx z23, z31
famin   z23.h, p3/m, z23.h, z13.h  // 01100101-01001111-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: famin   z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x4f,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 654f8db7 <unknown>

famin   z31.h, p7/m, z31.h, z30.h  // 01100101-01001111-10011111-11011111
// CHECK-INST: famin   z31.h, p7/m, z31.h, z30.h
// CHECK-ENCODING: [0xdf,0x9f,0x4f,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 654f9fdf <unknown>

famin   z0.s, p0/m, z0.s, z1.s  // 01100101-10001111-10000000-00100000
// CHECK-INST: famin   z0.s, p0/m, z0.s, z1.s
// CHECK-ENCODING: [0x20,0x80,0x8f,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 658f8020 <unknown>

movprfx z23, z31
famin   z23.s, p3/m, z23.s, z13.s  // 01100101-10001111-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: famin   z23.s, p3/m, z23.s, z13.s
// CHECK-ENCODING: [0xb7,0x8d,0x8f,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 658f8db7 <unknown>

famin   z31.s, p7/m, z31.s, z30.s  // 01100101-10001111-10011111-11011111
// CHECK-INST: famin   z31.s, p7/m, z31.s, z30.s
// CHECK-ENCODING: [0xdf,0x9f,0x8f,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 658f9fdf <unknown>

famin   z0.d, p0/m, z0.d, z1.d  // 01100101-11001111-10000000-00100000
// CHECK-INST: famin   z0.d, p0/m, z0.d, z1.d
// CHECK-ENCODING: [0x20,0x80,0xcf,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 65cf8020 <unknown>

movprfx z23, z31
famin   z23.d, p3/m, z23.d, z13.d  // 01100101-11001111-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: famin   z23.d, p3/m, z23.d, z13.d
// CHECK-ENCODING: [0xb7,0x8d,0xcf,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 65cf8db7 <unknown>

famin   z31.d, p7/m, z31.d, z30.d  // 01100101-11001111-10011111-11011111
// CHECK-INST: famin   z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xcf,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 65cf9fdf <unknown>


// FAMAX

famax   z0.h, p0/m, z0.h, z1.h  // 01100101-01001110-10000000-00100000
// CHECK-INST: famax   z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0x80,0x4e,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 654e8020 <unknown>

movprfx z23, z31
famax   z23.h, p3/m, z23.h, z13.h  // 01100101-01001110-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: famax   z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x4e,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 654e8db7 <unknown>

famax   z31.h, p7/m, z31.h, z30.h  // 01100101-01001110-10011111-11011111
// CHECK-INST: famax   z31.h, p7/m, z31.h, z30.h
// CHECK-ENCODING: [0xdf,0x9f,0x4e,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 654e9fdf <unknown>

famax   z0.s, p0/m, z0.s, z1.s  // 01100101-10001110-10000000-00100000
// CHECK-INST: famax   z0.s, p0/m, z0.s, z1.s
// CHECK-ENCODING: [0x20,0x80,0x8e,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 658e8020 <unknown>

movprfx z23, z31
famax   z23.s, p3/m, z23.s, z13.s  // 01100101-10001110-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: famax   z23.s, p3/m, z23.s, z13.s
// CHECK-ENCODING: [0xb7,0x8d,0x8e,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 658e8db7 <unknown>

famax   z31.s, p7/m, z31.s, z30.s  // 01100101-10001110-10011111-11011111
// CHECK-INST: famax   z31.s, p7/m, z31.s, z30.s
// CHECK-ENCODING: [0xdf,0x9f,0x8e,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 658e9fdf <unknown>

famax   z0.d, p0/m, z0.d, z1.d  // 01100101-11001110-10000000-00100000
// CHECK-INST: famax   z0.d, p0/m, z0.d, z1.d
// CHECK-ENCODING: [0x20,0x80,0xce,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 65ce8020 <unknown>

movprfx z23, z31
famax   z23.d, p3/m, z23.d, z13.d  // 01100101-11001110-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: famax   z23.d, p3/m, z23.d, z13.d
// CHECK-ENCODING: [0xb7,0x8d,0xce,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 65ce8db7 <unknown>

famax   z31.d, p7/m, z31.d, z30.d  // 01100101-11001110-10011111-11011111
// CHECK-INST: famax   z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xce,0x65]
// CHECK-ERROR: instruction requires: faminmax sve2 or sme2
// CHECK-UNKNOWN: 65ce9fdf <unknown>