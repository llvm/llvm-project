// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p3 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

addqp z0.b, z0.b, z0.b
// CHECK-INST: addqp z0.b, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0x78,0x20,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04207800 <unknown>

addqp z31.b, z31.b, z31.b
// CHECK-INST: addqp z31.b, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0x7b,0x3f,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 043f7bff <unknown>

addqp z0.h, z0.h, z0.h
// CHECK-INST: addqp z0.h, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0x78,0x60,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04607800 <unknown>

addqp z31.h, z31.h, z31.h
// CHECK-INST: addqp z31.h, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0x7b,0x7f,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 047f7bff <unknown>

addqp z0.s, z0.s, z0.s
// CHECK-INST: addqp z0.s, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0x78,0xa0,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04a07800 <unknown>

addqp z31.s, z31.s, z31.s
// CHECK-INST: addqp z31.s, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0x7b,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04bf7bff <unknown>

addqp z0.d, z0.d, z0.d
// CHECK-INST: addqp z0.d, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0x78,0xe0,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04e07800 <unknown>

addqp z31.d, z31.d, z31.d
// CHECK-INST: addqp z31.d, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0x7b,0xff,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04ff7bff <unknown>

// --------------------------------------------------------------------------//
// Test addsubp

addsubp z0.b, z0.b, z0.b
// CHECK-INST: addsubp z0.b, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0x7c,0x20,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04207c00 <unknown>

addsubp z31.b, z31.b, z31.b
// CHECK-INST: addsubp z31.b, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0x7f,0x3f,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 043f7fff <unknown>

addsubp z0.h, z0.h, z0.h
// CHECK-INST: addsubp z0.h, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0x7c,0x60,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04607c00 <unknown>

addsubp z31.h, z31.h, z31.h
// CHECK-INST: addsubp z31.h, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0x7f,0x7f,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 047f7fff <unknown>

addsubp z0.s, z0.s, z0.s
// CHECK-INST: addsubp z0.s, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0x7c,0xa0,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04a07c00 <unknown>

addsubp z31.s, z31.s, z31.s
// CHECK-INST: addsubp z31.s, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0x7f,0xbf,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04bf7fff <unknown>

addsubp z0.d, z0.d, z0.d
// CHECK-INST: addsubp z0.d, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0x7c,0xe0,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04e07c00 <unknown>

addsubp z31.d, z31.d, z31.d
// CHECK-INST: addsubp z31.d, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0x7f,0xff,0x04]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 04ff7fff <unknown>

// --------------------------------------------------------------------------//
// Test sabal

sabal z0.h, z0.b, z0.b
// CHECK-INST: sabal z0.h, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0xd4,0x40,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4440d400 <unknown>

sabal z31.h, z31.b, z31.b
// CHECK-INST: sabal z31.h, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0xd7,0x5f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 445fd7ff <unknown>

sabal z0.s, z0.h, z0.h
// CHECK-INST: sabal z0.s, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xd4,0x80,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4480d400 <unknown>

sabal z31.s, z31.h, z31.h
// CHECK-INST: sabal z31.s, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xd7,0x9f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 449fd7ff <unknown>

sabal z0.d, z0.s, z0.s
// CHECK-INST: sabal z0.d, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xd4,0xc0,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44c0d400 <unknown>

sabal z31.d, z31.s, z31.s
// CHECK-INST: sabal z31.d, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xd7,0xdf,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44dfd7ff <unknown>

movprfx z0, z7
sabal	z0.h, z1.b, z2.b
// CHECK-INST: movprfx z0, z7
// CHECK-INST: sabal z0.h, z1.b, z2.b
// CHECK-ENCODING: encoding: [0x20,0xd4,0x42,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4442d420 <unknown>

// --------------------------------------------------------------------------//
// Test uabal

uabal z0.h, z0.b, z0.b
// CHECK-INST: uabal z0.h, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0xdc,0x40,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4440dc00 <unknown>

uabal z31.h, z31.b, z31.b
// CHECK-INST: uabal z31.h, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0xdf,0x5f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 445fdfff <unknown>

uabal z0.s, z0.h, z0.h
// CHECK-INST: uabal z0.s, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xdc,0x80,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4480dc00 <unknown>

uabal z31.s, z31.h, z31.h
// CHECK-INST: uabal z31.s, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xdf,0x9f,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 449fdfff <unknown>

uabal z0.d, z0.s, z0.s
// CHECK-INST: uabal z0.d, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xdc,0xc0,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44c0dc00 <unknown>

uabal z31.d, z31.s, z31.s
// CHECK-INST: uabal z31.d, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xdf,0xdf,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44dfdfff <unknown>

movprfx z0, z7
uabal	z0.h, z1.b, z2.b
// CHECK-INST: movprfx z0, z7
// CHECK-INST: uabal z0.h, z1.b, z2.b
// CHECK-ENCODING: encoding: [0x20,0xdc,0x42,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4442dc20 <unknown>

// --------------------------------------------------------------------------//
// Test subp

subp z0.b, p0/m, z0.b, z0.b
// CHECK-INST: subp z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: encoding: [0x00,0xa0,0x10,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4410a000 <unknown>

subp z31.b, p7/m, z31.b, z31.b
// CHECK-INST: subp z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xff,0xbf,0x10,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4410bfff <unknown>

subp z0.h, p0/m, z0.h, z0.h
// CHECK-INST: subp z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xa0,0x50,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4450a000 <unknown>

subp z31.h, p7/m, z31.h, z31.h
// CHECK-INST: subp z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xbf,0x50,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4450bfff <unknown>

subp z0.s, p0/m, z0.s, z0.s
// CHECK-INST: subp z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xa0,0x90,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4490a000 <unknown>

subp z31.s, p7/m, z31.s, z31.s
// CHECK-INST: subp z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xbf,0x90,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4490bfff <unknown>

subp z0.d, p0/m, z0.d, z0.d
// CHECK-INST: subp z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xa0,0xd0,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44d0a000 <unknown>

subp z31.d, p7/m, z31.d, z31.d
// CHECK-INST: subp z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xbf,0xd0,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 44d0bfff <unknown>

movprfx z0.b, p0/m, z7.b
subp	z0.b, p0/m, z0.b, z1.b
// CHECK-INST: movprfx z0.b, p0/m, z7.b
// CHECK-INST: subp z0.b, p0/m, z0.b, z1.b
// CHECK-ENCODING: encoding: [0x20,0xa0,0x10,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4410a020 <unknown>

movprfx z0, z7
subp	z0.b, p0/m, z0.b, z1.b
// CHECK-INST: movprfx z0, z7
// CHECK-INST: subp z0.b, p0/m, z0.b, z1.b
// CHECK-ENCODING: encoding: [0x20,0xa0,0x10,0x44]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4410a020 <unknown>
