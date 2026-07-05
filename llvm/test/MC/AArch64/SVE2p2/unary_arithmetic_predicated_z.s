// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// ABS

abs     z0.b, p0/z, z0.b  // 00000100-00000110-10100000-00000000
// CHECK-INST: abs     z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x06,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0406a000 <unknown>

abs     z31.d, p7/z, z31.d  // 00000100-11000110-10111111-11111111
// CHECK-INST: abs     z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc6,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c6bfff <unknown>


// CLS

cls     z0.b, p0/z, z0.b  // 00000100-00001000-10100000-00000000
// CHECK-INST: cls     z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x08,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0408a000 <unknown>

clz     z31.d, p7/z, z31.d  // 00000100-11001001-10111111-11111111
// CHECK-INST: clz     z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc9,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c9bfff <unknown>

// CLZ

clz     z0.b, p0/z, z0.b  // 00000100-00001001-10100000-00000000
// CHECK-INST: clz     z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x09,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0409a000 <unknown>

clz     z31.d, p7/z, z31.d  // 00000100-11001001-10111111-11111111
// CHECK-INST: clz     z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc9,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c9bfff <unknown>

// CNOT

cnot    z0.b, p0/z, z0.b  // 00000100-00001011-10100000-00000000
// CHECK-INST: cnot    z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x0b,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 040ba000 <unknown>

cnot    z31.d, p7/z, z31.d  // 00000100-11001011-10111111-11111111
// CHECK-INST: cnot    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xcb,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04cbbfff <unknown>

// CNT

cnt     z0.b, p0/z, z0.b  // 00000100-00001010-10100000-00000000
// CHECK-INST: cnt     z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x0a,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 040aa000 <unknown>

cnt     z31.d, p7/z, z31.d  // 00000100-11001010-10111111-11111111
// CHECK-INST: cnt     z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xca,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04cabfff <unknown>


// FABS

fabs    z0.h, p0/z, z0.h  // 00000100-01001100-10100000-00000000
// CHECK-INST: fabs    z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x4c,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 044ca000 <unknown>

fabs    z31.d, p7/z, z31.d  // 00000100-11001100-10111111-11111111
// CHECK-INST: fabs    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xcc,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04ccbfff <unknown>

// FNEG

fneg    z0.h, p0/z, z0.h  // 00000100-01001101-10100000-00000000
// CHECK-INST: fneg    z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x4d,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 044da000 <unknown>

fneg    z31.d, p7/z, z31.d  // 00000100-11001101-10111111-11111111
// CHECK-INST: fneg    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xcd,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04cdbfff <unknown>

// NEG

neg     z0.b, p0/z, z0.b  // 00000100-00000111-10100000-00000000
// CHECK-INST: neg     z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x07,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0407a000 <unknown>

neg     z31.d, p7/z, z31.d  // 00000100-11000111-10111111-11111111
// CHECK-INST: neg     z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc7,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c7bfff <unknown>

//NOT

not     z0.b, p0/z, z0.b  // 00000100-00001110-10100000-00000000
// CHECK-INST: not     z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x0e,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 040ea000 <unknown>

not     z31.d, p7/z, z31.d  // 00000100-11001110-10111111-11111111
// CHECK-INST: not     z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xce,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04cebfff <unknown>

// SXTB

sxtb    z0.h, p0/z, z0.h  // 00000100-01000000-10100000-00000000
// CHECK-INST: sxtb    z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x40,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0440a000 <unknown>

sxtb    z31.d, p7/z, z31.d  // 00000100-11000000-10111111-11111111
// CHECK-INST: sxtb    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc0,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c0bfff <unknown>

// SXTH

sxth    z0.s, p0/z, z0.s  // 00000100-10000010-10100000-00000000
// CHECK-INST: sxth    z0.s, p0/z, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x82,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0482a000 <unknown>

sxth    z31.d, p7/z, z31.d  // 00000100-11000010-10111111-11111111
// CHECK-INST: sxth    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc2,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c2bfff <unknown>

// SXTW

sxtw    z0.d, p0/z, z0.d  // 00000100-11000100-10100000-00000000
// CHECK-INST: sxtw    z0.d, p0/z, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xc4,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c4a000 <unknown>

sxtw    z31.d, p7/z, z31.d  // 00000100-11000100-10111111-11111111
// CHECK-INST: sxtw    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc4,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c4bfff <unknown>

// UXTB

uxtb    z0.h, p0/z, z0.h  // 00000100-01000001-10100000-00000000
// CHECK-INST: uxtb    z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x41,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0441a000 <unknown>

uxtb    z31.d, p7/z, z31.d  // 00000100-11000001-10111111-11111111
// CHECK-INST: uxtb    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc1,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c1bfff <unknown>

uxth    z0.s, p0/z, z0.s  // 00000100-10000011-10100000-00000000
// CHECK-INST: uxth    z0.s, p0/z, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x83,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 0483a000 <unknown>

uxth    z31.d, p7/z, z31.d  // 00000100-11000011-10111111-11111111
// CHECK-INST: uxth    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc3,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c3bfff <unknown>

// UXTW

uxtw    z0.d, p0/z, z0.d  // 00000100-11000101-10100000-00000000
// CHECK-INST: uxtw    z0.d, p0/z, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xc5,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c5a000 <unknown>

uxtw    z31.d, p7/z, z31.d  // 00000100-11000101-10111111-11111111
// CHECK-INST: uxtw    z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc5,0x04]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 04c5bfff <unknown>
