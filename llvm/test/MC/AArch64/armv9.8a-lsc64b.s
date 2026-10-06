// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lsc64b < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lsc64b < %s \
// RUN:        | llvm-objdump -d --mattr=+lsc64b --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lsc64b < %s \
// RUN:        | llvm-objdump -d --mattr=-lsc64b --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lsc64b < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+lsc64b -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

lda64b x0, [x13]
// CHECK-INST: lda64b x0, [x13]
// CHECK-ENCODING: encoding: [0xa0,0xd1,0xbf,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f8bfd1a0 <unknown>

lda64b x2, [x13, #0]
// CHECK-INST: lda64b x2, [x13]
// CHECK-ENCODING: encoding: [0xa2,0xd1,0xbf,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f8bfd1a2 <unknown>

lda64b x0, [sp]
// CHECK-INST: lda64b x0, [sp]
// CHECK-ENCODING: encoding: [0xe0,0xd3,0xbf,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f8bfd3e0 <unknown>

lda64b x0, [sp, #0]
// CHECK-INST: lda64b x0, [sp]
// CHECK-ENCODING: encoding: [0xe0,0xd3,0xbf,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f8bfd3e0 <unknown>

stl64b x14, [x13]
// CHECK-INST: stl64b x14, [x13]
// CHECK-ENCODING: encoding: [0xae,0x91,0x7f,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f87f91ae <unknown>

stl64b x14, [sp, #0]
// CHECK-INST: stl64b x14, [sp]
// CHECK-ENCODING: encoding: [0xee,0x93,0x7f,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f87f93ee <unknown>

stl64b x14, [x13, #0]
// CHECK-INST: stl64b x14, [x13]
// CHECK-ENCODING: encoding: [0xae,0x91,0x7f,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f87f91ae <unknown>

stl64bv x1, x20, [x13]
// CHECK-INST: stl64bv x1, x20, [x13]
// CHECK-ENCODING: encoding: [0xb4,0xb1,0x61,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f861b1b4 <unknown>

stl64bv x1, x20, [sp]
// CHECK-INST: stl64bv x1, x20, [sp]
// CHECK-ENCODING: encoding: [0xf4,0xb3,0x61,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f861b3f4 <unknown>

stl64bv0 x1, x22, [x13]
// CHECK-INST: stl64bv0 x1, x22, [x13]
// CHECK-ENCODING: encoding: [0xb6,0xa1,0x61,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f861a1b6 <unknown>

stl64bv0 x1, x22, [sp]
// CHECK-INST: stl64bv0 x1, x22, [sp]
// CHECK-ENCODING: encoding: [0xf6,0xa3,0x61,0xf8]
// CHECK-ERROR: error: instruction requires: lsc64b
// CHECK-UNKNOWN: f861a3f6 <unknown>
