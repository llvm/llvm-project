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

str     za[w12, 0], [x0]
// CHECK-INST: str     za[w12, 0], [x0]
// CHECK-ENCODING: [0x00,0x00,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 00 00 20 e1 <unknown>

str     za[w14, 5], [x10, #5, mul vl]
// CHECK-INST: str     za[w14, 5], [x10, #5, mul vl]
// CHECK-ENCODING: [0x45,0x41,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 45 41 20 e1 <unknown>

str     za[w15, 7], [x13, #7, mul vl]
// CHECK-INST: str     za[w15, 7], [x13, #7, mul vl]
// CHECK-ENCODING: [0xa7,0x61,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a7 61 20 e1 <unknown>

str     za[w15, 15], [sp, #15, mul vl]
// CHECK-INST: str     za[w15, 15], [sp, #15, mul vl]
// CHECK-ENCODING: [0xef,0x63,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: ef 63 20 e1 <unknown>

str     za[w12, 5], [x17, #5, mul vl]
// CHECK-INST: str     za[w12, 5], [x17, #5, mul vl]
// CHECK-ENCODING: [0x25,0x02,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 25 02 20 e1 <unknown>

str     za[w12, 1], [x1, #1, mul vl]
// CHECK-INST: str     za[w12, 1], [x1, #1, mul vl]
// CHECK-ENCODING: [0x21,0x00,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 00 20 e1 <unknown>

str     za[w14, 8], [x19, #8, mul vl]
// CHECK-INST: str     za[w14, 8], [x19, #8, mul vl]
// CHECK-ENCODING: [0x68,0x42,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 68 42 20 e1 <unknown>

str     za[w12, 0], [x12]
// CHECK-INST: str     za[w12, 0], [x12]
// CHECK-ENCODING: [0x80,0x01,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 80 01 20 e1 <unknown>

str     za[w14, 1], [x1, #1, mul vl]
// CHECK-INST: str     za[w14, 1], [x1, #1, mul vl]
// CHECK-ENCODING: [0x21,0x40,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 21 40 20 e1 <unknown>

str     za[w12, 13], [x22, #13, mul vl]
// CHECK-INST: str     za[w12, 13], [x22, #13, mul vl]
// CHECK-ENCODING: [0xcd,0x02,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: cd 02 20 e1 <unknown>

str     za[w15, 2], [x9, #2, mul vl]
// CHECK-INST: str     za[w15, 2], [x9, #2, mul vl]
// CHECK-ENCODING: [0x22,0x61,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 22 61 20 e1 <unknown>

str     za[w13, 7], [x12, #7, mul vl]
// CHECK-INST: str     za[w13, 7], [x12, #7, mul vl]
// CHECK-ENCODING: [0x87,0x21,0x20,0xe1]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 87 21 20 e1 <unknown>
