// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// --------------------------------------------------------------------------//
// 8-bit

sclamp  z0.b, z0.b, z0.b
// CHECK-INST: sclamp  z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xc0,0x00,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4400c000 <unknown>

sclamp  z21.b, z10.b, z21.b
// CHECK-INST: sclamp  z21.b, z10.b, z21.b
// CHECK-ENCODING: [0x55,0xc1,0x15,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4415c155 <unknown>

sclamp  z23.b, z13.b, z8.b
// CHECK-INST: sclamp  z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xc1,0x08,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4408c1b7 <unknown>

sclamp  z31.b, z31.b, z31.b
// CHECK-INST: sclamp  z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xc3,0x1f,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 441fc3ff <unknown>

// --------------------------------------------------------------------------//
// 16-bit

sclamp  z0.h, z0.h, z0.h
// CHECK-INST: sclamp  z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc0,0x40,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4440c000 <unknown>

sclamp  z21.h, z10.h, z21.h
// CHECK-INST: sclamp  z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xc1,0x55,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4455c155 <unknown>

sclamp  z23.h, z13.h, z8.h
// CHECK-INST: sclamp  z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xc1,0x48,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4448c1b7 <unknown>

sclamp  z31.h, z31.h, z31.h
// CHECK-INST: sclamp  z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xc3,0x5f,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 445fc3ff <unknown>

// --------------------------------------------------------------------------//
// 32-bit

sclamp  z0.s, z0.s, z0.s
// CHECK-INST: sclamp  z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xc0,0x80,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4480c000 <unknown>

sclamp  z21.s, z10.s, z21.s
// CHECK-INST: sclamp  z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0xc1,0x95,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4495c155 <unknown>

sclamp  z23.s, z13.s, z8.s
// CHECK-INST: sclamp  z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0xc1,0x88,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4488c1b7 <unknown>

sclamp  z31.s, z31.s, z31.s
// CHECK-INST: sclamp  z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0xc3,0x9f,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 449fc3ff <unknown>

// --------------------------------------------------------------------------//
// 64-bit

sclamp  z0.d, z0.d, z0.d
// CHECK-INST: sclamp  z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xc0,0xc0,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44c0c000 <unknown>

sclamp  z21.d, z10.d, z21.d
// CHECK-INST: sclamp  z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0xc1,0xd5,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44d5c155 <unknown>

sclamp  z23.d, z13.d, z8.d
// CHECK-INST: sclamp  z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0xc1,0xc8,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44c8c1b7 <unknown>

sclamp  z31.d, z31.d, z31.d
// CHECK-INST: sclamp  z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0xc3,0xdf,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44dfc3ff <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z23, z27
// CHECK-INST: movprfx  z23, z27
// CHECK-ENCODING: [0x77,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf77 <unknown>

sclamp  z23.b, z13.b, z8.b
// CHECK-INST: sclamp  z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xc1,0x08,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4408c1b7 <unknown>

movprfx z23, z27
// CHECK-INST: movprfx  z23, z27
// CHECK-ENCODING: [0x77,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf77 <unknown>

sclamp  z23.h, z13.h, z8.h
// CHECK-INST: sclamp  z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xc1,0x48,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4448c1b7 <unknown>

movprfx z23, z27
// CHECK-INST: movprfx  z23, z27
// CHECK-ENCODING: [0x77,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf77 <unknown>

sclamp  z23.s, z13.s, z8.s
// CHECK-INST: sclamp  z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0xc1,0x88,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 4488c1b7 <unknown>

movprfx z23, z27
// CHECK-INST: movprfx  z23, z27
// CHECK-ENCODING: [0x77,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf77 <unknown>

sclamp  z23.d, z13.d, z8.d
// CHECK-INST: sclamp  z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0xc1,0xc8,0x44]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 44c8c1b7 <unknown>
