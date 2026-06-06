// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+ssve-fp8fma < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8fma < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+fp8fma --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+fp8fma < %s \
// RUN:        | llvm-objdump -d --mattr=-fp8fma --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+fp8fma < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+fp8fma -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//
// FMLALB instructions
//
// fmlalb - indexed

fmlalb  z0.h, z0.b, z0.b[0]  // 01100100-00100000-01010000-00000000
// CHECK-INST: fmlalb  z0.h, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x50,0x20,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64205000 <unknown>

movprfx z23, z31
fmlalb  z23.h, z13.b, z0.b[7]  // 01100100-00101000-01011101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalb  z23.h, z13.b, z0.b[7]
// CHECK-ENCODING: [0xb7,0x5d,0x28,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64285db7 <unknown>

fmlalb  z31.h, z31.b, z7.b[15]  // 01100100-00111111-01011111-11111111
// CHECK-INST: fmlalb  z31.h, z31.b, z7.b[15]
// CHECK-ENCODING: [0xff,0x5f,0x3f,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 643f5fff <unknown>


// fmlalb - group

fmlalb  z0.h, z0.b, z0.b  // 01100100-10100000-10001000-00000000
// CHECK-INST: fmlalb  z0.h, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x88,0xa0,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a08800 <unknown>

movprfx z23, z31
fmlalb  z23.h, z13.b, z8.b  // 01100100-10101000-10001001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalb  z23.h, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x89,0xa8,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a889b7 <unknown>

fmlalb  z31.h, z31.b, z31.b  // 01100100-10111111-10001011-11111111
// CHECK-INST: fmlalb  z31.h, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x8b,0xbf,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64bf8bff <unknown>


//
// FMLALT instructions
//
// fmlalt - indexed

fmlalt  z0.h, z0.b, z0.b[0]  // 01100100-10100000-01010000-00000000
// CHECK-INST: fmlalt  z0.h, z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x50,0xa0,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a05000 <unknown>

movprfx z23, z31
fmlalt  z23.h, z13.b, z0.b[7]  // 01100100-10101000-01011101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalt  z23.h, z13.b, z0.b[7]
// CHECK-ENCODING: [0xb7,0x5d,0xa8,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a85db7 <unknown>

fmlalt  z31.h, z31.b, z7.b[15]  // 01100100-10111111-01011111-11111111
// CHECK-INST: fmlalt  z31.h, z31.b, z7.b[15]
// CHECK-ENCODING: [0xff,0x5f,0xbf,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64bf5fff <unknown>


// fmlalt - group

fmlalt  z0.h, z0.b, z0.b  // 01100100-10100000-10011000-00000000
// CHECK-INST: fmlalt  z0.h, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x98,0xa0,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a09800 <unknown>

movprfx z23, z31
fmlalt  z23.h, z13.b, z8.b  // 01100100-10101000-10011001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmlalt  z23.h, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x99,0xa8,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64a899b7 <unknown>

fmlalt  z31.h, z31.b, z31.b  // 01100100-10111111-10011011-11111111
// CHECK-INST: fmlalt  z31.h, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x9b,0xbf,0x64]
// CHECK-ERROR: instruction requires: ssve-fp8fma or (sve2 and fp8fma)
// CHECK-UNKNOWN: 64bf9bff <unknown>
