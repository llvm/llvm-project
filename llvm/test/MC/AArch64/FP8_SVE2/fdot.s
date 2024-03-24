// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+ssve-fp8dot2,+ssve-fp8dot4 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8dot2,+fp8dot4 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+fp8dot2,+fp8dot4 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8dot2,+fp8dot4 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8dot2,fp8dot4 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+fp8dot2,fp8dot4 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//
// FDOT2 instructions
//
// fdot2 - indexed

fdot    z0.h, z0.b, z0.b[0]  // 01100100-00100000-01000100-00000000
// CHECK-INST: fdot    z0.h, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x44,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot2 or (sve2 and fp8dot2)
// CHECK-UNKNOWN: 64204400 <unknown>

movprfx z23, z31
fdot    z23.h, z13.b, z0.b[3]  // 01100100-00101000-01001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fdot    z23.h, z13.b, z0.b[3]
// CHECK-ENCODING: [0xb7,0x4d,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot2 or (sve2 and fp8dot2)
// CHECK-UNKNOWN: 64284db7 <unknown>

fdot    z31.h, z31.b, z7.b[7]  // 01100100-00111111-01001111-11111111
// CHECK-INST: fdot    z31.h, z31.b, z7.b[7]
// CHECK-ENCODING: [0xff,0x4f,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot2 or (sve2 and fp8dot2)
// CHECK-UNKNOWN: 643f4fff <unknown>


// fdot2 - group

fdot    z0.h, z0.b, z0.b  // 01100100-00100000-10000100-00000000
// CHECK-INST: fdot    z0.h, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x84,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot2 or (sve2 and fp8dot2)
// CHECK-UNKNOWN: 64208400 <unknown>

movprfx z23, z31
fdot    z23.h, z13.b, z8.b  // 01100100-00101000-10000101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fdot    z23.h, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x85,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot2 or (sve2 and fp8dot2)
// CHECK-UNKNOWN: 642885b7 <unknown>

fdot    z31.h, z31.b, z31.b  // 01100100-00111111-10000111-11111111
// CHECK-INST: fdot    z31.h, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x87,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot2 or (sve2 and fp8dot2)
// CHECK-UNKNOWN: 643f87ff <unknown>


//
// FDOT4 instructions
//
// fdot4 - indexed

fdot    z0.s, z0.b, z0.b[0]  // 01100100-01100000-01000100-00000000
// CHECK-INST: fdot    z0.s, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x44,0x60,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot4 or (sve2 and fp8dot4)
// CHECK-UNKNOWN: 64604400 <unknown>

movprfx z23, z31
fdot    z23.s, z13.b, z0.b[1]  // 01100100-01101000-01000101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fdot    z23.s, z13.b, z0.b[1]
// CHECK-ENCODING: [0xb7,0x45,0x68,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot4 or (sve2 and fp8dot4)
// CHECK-UNKNOWN: 646845b7 <unknown>

fdot    z31.s, z31.b, z7.b[3]  // 01100100-01111111-01000111-11111111
// CHECK-INST: fdot    z31.s, z31.b, z7.b[3]
// CHECK-ENCODING: [0xff,0x47,0x7f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot4 or (sve2 and fp8dot4)
// CHECK-UNKNOWN: 647f47ff <unknown>

// fdot4 - group

fdot    z0.s, z0.b, z0.b  // 01100100-01100000-10000100-00000000
// CHECK-INST: fdot    z0.s, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x84,0x60,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot4 or (sve2 and fp8dot4)
// CHECK-UNKNOWN: 64608400 <unknown>

movprfx z23, z31
fdot    z23.s, z13.b, z8.b  // 01100100-01101000-10000101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fdot    z23.s, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x85,0x68,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot4 or (sve2 and fp8dot4)
// CHECK-UNKNOWN: 646885b7 <unknown>

fdot    z31.s, z31.b, z31.b  // 01100100-01111111-10000111-11111111
// CHECK-INST: fdot    z31.s, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x87,0x7f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8dot4 or (sve2 and fp8dot4)
// CHECK-UNKNOWN: 647f87ff <unknown>
