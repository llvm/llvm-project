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

// -----------------------------------------------------------------
// Signed saturating rounding shift right narrow by immediate and interleave

sqrshrn z0.b, { z0.h, z1.h }, #1
// CHECK-INST: sqrshrn z0.b, { z0.h, z1.h }, #1
// CHECK-ENCODING: encoding: [0x00,0x28,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af2800 <unknown>

sqrshrn z31.b, { z30.h, z31.h }, #1
// CHECK-INST: sqrshrn z31.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xdf,0x2b,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af2bdf <unknown>

sqrshrn z0.b, { z0.h, z1.h }, #8
// CHECK-INST: sqrshrn z0.b, { z0.h, z1.h }, #8
// CHECK-ENCODING: encoding: [0x00,0x28,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a82800 <unknown>

sqrshrn z31.b, { z30.h, z31.h }, #3
// CHECK-INST: sqrshrn z31.b, { z30.h, z31.h }, #3
// CHECK-ENCODING: encoding: [0xdf,0x2b,0xad,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45ad2bdf <unknown>

// -----------------------------------------------------------------
// Signed saturating rounding shift right unsigned narrow by immediate and interleave

sqrshrun z0.b, { z0.h, z1.h }, #1
// CHECK-INST: sqrshrun z0.b, { z0.h, z1.h }, #1
// CHECK-ENCODING: encoding: [0x00,0x08,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af0800 <unknown>

sqrshrun z31.b, { z30.h, z31.h }, #1
// CHECK-INST: sqrshrun z31.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xdf,0x0b,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af0bdf <unknown>

sqrshrun z0.b, { z0.h, z1.h }, #8
// CHECK-INST: sqrshrun z0.b, { z0.h, z1.h }, #8
// CHECK-ENCODING: encoding: [0x00,0x08,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a80800 <unknown>

sqrshrun z31.b, { z30.h, z31.h }, #8
// CHECK-INST: sqrshrun z31.b, { z30.h, z31.h }, #8
// CHECK-ENCODING: encoding: [0xdf,0x0b,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a80bdf <unknown>

// -----------------------------------------------------------------
// Signed saturating shift right narrow by immediate and interleave

sqshrn z21.b, { z30.h, z31.h }, #1
// CHECK-INST: sqshrn z21.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xd5,0x03,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af03d5 <unknown>

sqshrn z31.b, { z30.h, z31.h }, #1
// CHECK-INST: sqshrn z31.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xdf,0x03,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af03df <unknown>

sqshrn z10.b, { z0.h, z1.h }, #8
// CHECK-INST: sqshrn z10.b, { z0.h, z1.h }, #8
// CHECK-ENCODING: encoding: [0x0a,0x00,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a8000a <unknown>

sqshrn z31.b, { z30.h, z31.h }, #8
// CHECK-INST: sqshrn z31.b, { z30.h, z31.h }, #8
// CHECK-ENCODING: encoding: [0xdf,0x03,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a803df <unknown>

sqshrn z0.b, { z0.h, z1.h }, #1
// CHECK-INST: sqshrn z0.b, { z0.h, z1.h }, #1
// CHECK-ENCODING: encoding: [0x00,0x00,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af0000 <unknown>

sqshrn z0.h, { z0.s, z1.s }, #1
// CHECK-INST: sqshrn z0.h, { z0.s, z1.s }, #1
// CHECK-ENCODING: encoding: [0x00,0x00,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45bf0000 <unknown>

sqshrn z31.h, { z30.s, z31.s }, #1
// CHECK-INST: sqshrn z31.h, { z30.s, z31.s }, #1
// CHECK-ENCODING: encoding: [0xdf,0x03,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45bf03df <unknown>

sqshrn z0.h, { z0.s, z1.s }, #16
// CHECK-INST: sqshrn z0.h, { z0.s, z1.s }, #16
// CHECK-ENCODING: encoding: [0x00,0x00,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45b00000 <unknown>

sqshrn z31.h, { z30.s, z31.s }, #16
// CHECK-INST: sqshrn z31.h, { z30.s, z31.s }, #16
// CHECK-ENCODING: encoding: [0xdf,0x03,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45b003df <unknown>

// -----------------------------------------------------------------
// Signed saturating shift right unsigned narrow by immediate and interleave

sqshrun z0.b, { z0.h, z1.h }, #1
// CHECK-INST: sqshrun z0.b, { z0.h, z1.h }, #1
// CHECK-ENCODING: encoding: [0x00,0x20,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af2000 <unknown>

sqshrun z31.b, { z30.h, z31.h }, #1
// CHECK-INST: sqshrun z31.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xdf,0x23,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af23df <unknown>

sqshrun z0.b, { z0.h, z1.h }, #8
// CHECK-INST: sqshrun z0.b, { z0.h, z1.h }, #8
// CHECK-ENCODING: encoding: [0x00,0x20,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a82000 <unknown>

sqshrun z31.b, { z30.h, z31.h }, #8
// CHECK-INST: sqshrun z31.b, { z30.h, z31.h }, #8
// CHECK-ENCODING: encoding: [0xdf,0x23,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a823df <unknown>

sqshrun z0.h, { z0.s, z1.s }, #1
// CHECK-INST: sqshrun z0.h, { z0.s, z1.s }, #1
// CHECK-ENCODING: encoding: [0x00,0x20,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45bf2000 <unknown>

sqshrun z31.h, { z30.s, z31.s }, #1
// CHECK-INST: sqshrun z31.h, { z30.s, z31.s }, #1
// CHECK-ENCODING: encoding: [0xdf,0x23,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45bf23df <unknown>

sqshrun z0.h, { z0.s, z1.s }, #16
// CHECK-INST: sqshrun z0.h, { z0.s, z1.s }, #16
// CHECK-ENCODING: encoding: [0x00,0x20,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45b02000 <unknown>

sqshrun z31.h, { z30.s, z31.s }, #16
// CHECK-INST: sqshrun z31.h, { z30.s, z31.s }, #16
// CHECK-ENCODING: encoding: [0xdf,0x23,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45b023df <unknown>

// -----------------------------------------------------------------
// Unsigned saturating rounding shift right narrow by immediate and interleave

uqrshrn z0.b, { z0.h, z1.h }, #1
// CHECK-INST: uqrshrn z0.b, { z0.h, z1.h }, #1
// CHECK-ENCODING: encoding: [0x00,0x38,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af3800 <unknown>

uqrshrn z31.b, { z30.h, z31.h }, #1
// CHECK-INST: uqrshrn z31.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xdf,0x3b,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af3bdf <unknown>

uqrshrn z0.b, { z0.h, z1.h }, #8
// CHECK-INST: uqrshrn z0.b, { z0.h, z1.h }, #8
// CHECK-ENCODING: encoding: [0x00,0x38,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a83800 <unknown>

uqrshrn z31.b, { z30.h, z31.h }, #8
// CHECK-INST: uqrshrn z31.b, { z30.h, z31.h }, #8
// CHECK-ENCODING: encoding: [0xdf,0x3b,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a83bdf <unknown>

// -----------------------------------------------------------------
// Unsigned saturating shift right narrow by immediate and interleave

uqshrn z0.b, { z0.h, z1.h }, #1
// CHECK-INST: uqshrn z0.b, { z0.h, z1.h }, #1
// CHECK-ENCODING: encoding: [0x00,0x10,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af1000 <unknown>

uqshrn z31.b, { z30.h, z31.h }, #1
// CHECK-INST: uqshrn z31.b, { z30.h, z31.h }, #1
// CHECK-ENCODING: encoding: [0xdf,0x13,0xaf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45af13df <unknown>

uqshrn z0.b, { z0.h, z1.h }, #8
// CHECK-INST: uqshrn z0.b, { z0.h, z1.h }, #8
// CHECK-ENCODING: encoding: [0x00,0x10,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a81000 <unknown>

uqshrn z31.b, { z30.h, z31.h }, #8
// CHECK-INST: uqshrn z31.b, { z30.h, z31.h }, #8
// CHECK-ENCODING: encoding: [0xdf,0x13,0xa8,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45a813df <unknown>

uqshrn z0.h, { z0.s, z1.s }, #1
// CHECK-INST: uqshrn z0.h, { z0.s, z1.s }, #1
// CHECK-ENCODING: encoding: [0x00,0x10,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45bf1000 <unknown>

uqshrn z31.h, { z30.s, z31.s }, #1
// CHECK-INST: uqshrn z31.h, { z30.s, z31.s }, #1
// CHECK-ENCODING: encoding: [0xdf,0x13,0xbf,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45bf13df <unknown>

uqshrn z0.h, { z0.s, z1.s }, #16
// CHECK-INST: uqshrn z0.h, { z0.s, z1.s }, #16
// CHECK-ENCODING: encoding: [0x00,0x10,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45b01000 <unknown>

uqshrn z31.h, { z30.s, z31.s }, #16
// CHECK-INST: uqshrn z31.h, { z30.s, z31.s }, #16
// CHECK-ENCODING: encoding: [0xdf,0x13,0xb0,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45b013df <unknown>
