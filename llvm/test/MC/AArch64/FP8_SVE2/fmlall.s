// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+ssve-fp8fma < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8fma < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+fp8fma --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8fma < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8fma < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+fp8fma -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//
// FMLALLBB instructions
//
// fmlallbb - indexed

fmlallbb z0.s, z0.b, z0.b[0]  // 01100100-00100000-11000000-00000000
// CHECK-INST: fmlallbb z0.s, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0xc0,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6420c000 <unknown>

movprfx z23, z31
fmlallbb z23.s, z13.b, z0.b[7]  // 01100100-00101000-11001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlallbb z23.s, z13.b, z0.b[7]
// CHECK-ENCODING: [0xb7,0xcd,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6428cdb7 <unknown>

fmlallbb z31.s, z31.b, z7.b[15]  // 01100100-00111111-11001111-11111111
// CHECK-INST: fmlallbb z31.s, z31.b, z7.b[15]
// CHECK-ENCODING: [0xff,0xcf,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 643fcfff <unknown>

//
// FMLALLBB instructions
//
// fmlallbb - group

fmlallbb z0.s, z0.b, z0.b  // 01100100-00100000-10001000-00000000
// CHECK-INST: fmlallbb z0.s, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x88,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64208800 <unknown>

movprfx z23, z31
fmlallbb z23.s, z13.b, z8.b  // 01100100-00101000-10001001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlallbb z23.s, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x89,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 642889b7 <unknown>

fmlallbb z31.s, z31.b, z31.b  // 01100100-00111111-10001011-11111111
// CHECK-INST: fmlallbb z31.s, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x8b,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 643f8bff <unknown>

//--------------------------------------------//
//
// FMLALLBT instructions
//
// fmlallbt - indexed

fmlallbt z0.s, z0.b, z0.b[0]  // 01100100-01100000-11000000-00000000
// CHECK-INST: fmlallbt z0.s, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0xc0,0x60,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6460c000 <unknown>

movprfx z23, z31
fmlallbt z23.s, z13.b, z0.b[7]  // 01100100-01101000-11001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlallbt z23.s, z13.b, z0.b[7]
// CHECK-ENCODING: [0xb7,0xcd,0x68,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6468cdb7 <unknown>

fmlallbt z31.s, z31.b, z7.b[15]  // 01100100-01111111-11001111-11111111
// CHECK-INST: fmlallbt z31.s, z31.b, z7.b[15]
// CHECK-ENCODING: [0xff,0xcf,0x7f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 647fcfff <unknown>

//
// FMLALLBT instructions
//
// fmlallbt - group

fmlallbt z0.s, z0.b, z0.b  // 01100100-00100000-10011000-00000000
// CHECK-INST: fmlallbt z0.s, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x98,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64209800 <unknown>

movprfx z23, z31
fmlallbt z23.s, z13.b, z8.b  // 01100100-00101000-10011001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlallbt z23.s, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x99,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 642899b7 <unknown>

fmlallbt z31.s, z31.b, z31.b  // 01100100-00111111-10011011-11111111
// CHECK-INST: fmlallbt z31.s, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x9b,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 643f9bff <unknown>

//--------------------------------------------//
//
// FMLALLTB instructions
//
// fmlalltb - indexed

fmlalltb z0.s, z0.b, z0.b[0]  // 01100100-10100000-11000000-00000000
// CHECK-INST: fmlalltb z0.s, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a0c000 <unknown>

movprfx z23, z31
fmlalltb z23.s, z13.b, z0.b[7]  // 01100100-10101000-11001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalltb z23.s, z13.b, z0.b[7]
// CHECK-ENCODING: [0xb7,0xcd,0xa8,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a8cdb7 <unknown>

fmlalltb z31.s, z31.b, z7.b[15]  // 01100100-10111111-11001111-11111111
// CHECK-INST: fmlalltb z31.s, z31.b, z7.b[15]
// CHECK-ENCODING: [0xff,0xcf,0xbf,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64bfcfff <unknown>

//
// FMLALLTB instructions
//
// fmlalltb - group

fmlalltb z0.s, z0.b, z0.b  // 01100100-00100000-10101000-00000000
// CHECK-INST: fmlalltb z0.s, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xa8,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6420a800 <unknown>

movprfx z23, z31
fmlalltb z23.s, z13.b, z8.b  // 01100100-00101000-10101001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalltb z23.s, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xa9,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6428a9b7 <unknown>

fmlalltb z31.s, z31.b, z31.b  // 01100100-00111111-10101011-11111111
// CHECK-INST: fmlalltb z31.s, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xab,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 643fabff <unknown>


//--------------------------------------------//
//
// FMLALLTT instructions
//
// fmlalltt - indexed

fmlalltt z0.s, z0.b, z0.b[0]  // 01100100-11100000-11000000-00000000
// CHECK-INST: fmlalltt z0.s, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0xc0,0xe0,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64e0c000 <unknown>

movprfx z23, z31
fmlalltt z23.s, z13.b, z0.b[7]  // 01100100-11101000-11001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalltt z23.s, z13.b, z0.b[7]
// CHECK-ENCODING: [0xb7,0xcd,0xe8,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64e8cdb7 <unknown>

fmlalltt z31.s, z31.b, z7.b[15]  // 01100100-11111111-11001111-11111111
// CHECK-INST: fmlalltt z31.s, z31.b, z7.b[15]
// CHECK-ENCODING: [0xff,0xcf,0xff,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64ffcfff <unknown>


//
// FMLALLTT instructions
//
// fmlalltt - group

fmlalltt z0.s, z0.b, z0.b  // 01100100-00100000-10111000-00000000
// CHECK-INST: fmlalltt z0.s, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xb8,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6420b800 <unknown>

movprfx z23, z31
fmlalltt z23.s, z13.b, z8.b  // 01100100-00101000-10111001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalltt z23.s, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xb9,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 6428b9b7 <unknown>

fmlalltt z31.s, z31.b, z31.b  // 01100100-00111111-10111011-11111111
// CHECK-INST: fmlalltt z31.s, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xbb,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 643fbbff <unknown>
