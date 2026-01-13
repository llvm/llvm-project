// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+rcpc3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+rcpc3,+v8.9a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+rcpc3,+v9.4a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+rcpc3 < %s \
// RUN:        | llvm-objdump -d --mattr=+rcpc3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+rcpc3 < %s \
// RUN:   | llvm-objdump -d --mattr=-rcpc3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+rcpc3 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+rcpc3 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


stilp   w24, w0, [x16, #-8]!
// CHECK-INST: stilp w24, w0, [x16, #-8]!
// CHECK-ENCODING: encoding: [0x18,0x0a,0x00,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99000a18      <unknown>

stilp   w24, w0, [x16,  -8]!
// CHECK-INST: stilp w24, w0, [x16, #-8]!
// CHECK-ENCODING: encoding: [0x18,0x0a,0x00,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99000a18      <unknown>

stilp   x25, x1, [x17,  -16]!
// CHECK-INST: stilp x25, x1, [x17, #-16]!
// CHECK-ENCODING: encoding: [0x39,0x0a,0x01,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9010a39      <unknown>

stilp   x25, x1, [x17, #-16]!
// CHECK-INST: stilp x25, x1, [x17, #-16]!
// CHECK-ENCODING: encoding: [0x39,0x0a,0x01,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9010a39      <unknown>

stilp   w26, w2, [x18]
// CHECK-INST: stilp w26, w2, [x18]
// CHECK-ENCODING: encoding: [0x5a,0x1a,0x02,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99021a5a      <unknown>

stilp   w26, w2, [x18, #0]
// CHECK-INST: stilp w26, w2, [x18]
// CHECK-ENCODING: encoding: [0x5a,0x1a,0x02,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99021a5a      <unknown>

stilp   x27, x3, [sp]
// CHECK-INST: stilp x27, x3, [sp]
// CHECK-ENCODING: encoding: [0xfb,0x1b,0x03,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9031bfb      <unknown>

stilp   x27, x3, [sp, 0]
// CHECK-INST: stilp x27, x3, [sp]
// CHECK-ENCODING: encoding: [0xfb,0x1b,0x03,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9031bfb      <unknown>

ldiapp  w28, w4, [x20], #8
// CHECK-INST: ldiapp w28, w4, [x20], #8
// CHECK-ENCODING: encoding: [0x9c,0x0a,0x44,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99440a9c      <unknown>

ldiapp  w28, w4, [x20, #0], #8
// CHECK-INST: ldiapp w28, w4, [x20], #8
// CHECK-ENCODING: encoding: [0x9c,0x0a,0x44,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99440a9c      <unknown>

ldiapp  w28, w4, [x20],  8
// CHECK-INST: ldiapp w28, w4, [x20], #8
// CHECK-ENCODING: encoding: [0x9c,0x0a,0x44,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99440a9c      <unknown>

ldiapp  w28, w4, [x20, 0],  8
// CHECK-INST: ldiapp w28, w4, [x20], #8
// CHECK-ENCODING: encoding: [0x9c,0x0a,0x44,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99440a9c      <unknown>

ldiapp  x29, x5, [x21], #16
// CHECK-INST: ldiapp x29, x5, [x21], #16
// CHECK-ENCODING: encoding: [0xbd,0x0a,0x45,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9450abd      <unknown>

ldiapp  x29, x5, [x21],  16
// CHECK-INST: ldiapp x29, x5, [x21], #16
// CHECK-ENCODING: encoding: [0xbd,0x0a,0x45,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9450abd      <unknown>

ldiapp  w30, w6, [sp]
// CHECK-INST: ldiapp w30, w6, [sp]
// CHECK-ENCODING: encoding: [0xfe,0x1b,0x46,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99461bfe      <unknown>

ldiapp  w30, w6, [sp, #0]
// CHECK-INST: ldiapp w30, w6, [sp]
// CHECK-ENCODING: encoding: [0xfe,0x1b,0x46,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99461bfe      <unknown>

ldiapp  xzr, x7, [x23]
// CHECK-INST: ldiapp xzr, x7, [x23]
// CHECK-ENCODING: encoding: [0xff,0x1a,0x47,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9471aff      <unknown>

ldiapp  xzr, x7, [x23, 0]
// CHECK-INST: ldiapp xzr, x7, [x23]
// CHECK-ENCODING: encoding: [0xff,0x1a,0x47,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9471aff      <unknown>

stlr w3, [x15, #-4]!
// CHECK-INST: stlr w3, [x15, #-4]!
// CHECK-ENCODING: encoding: [0xe3,0x09,0x80,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  998009e3      <unknown>

stlr w3, [x15,  -4]!
// CHECK-INST: stlr w3, [x15, #-4]!
// CHECK-ENCODING: encoding: [0xe3,0x09,0x80,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  998009e3      <unknown>

stlr x3, [x15, #-8]!
// CHECK-INST: stlr x3, [x15, #-8]!
// CHECK-ENCODING: encoding: [0xe3,0x09,0x80,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d98009e3      <unknown>

stlr x3, [sp,  -8]!
// CHECK-INST: stlr x3, [sp, #-8]!
// CHECK-ENCODING: encoding: [0xe3,0x0b,0x80,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9800be3      <unknown>

ldapr w3, [sp], #4
// CHECK-INST: ldapr w3, [sp], #4
// CHECK-ENCODING: encoding: [0xe3,0x0b,0xc0,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99c00be3      <unknown>

ldapr w3, [x15], 4
// CHECK-INST: ldapr w3, [x15], #4
// CHECK-ENCODING: encoding: [0xe3,0x09,0xc0,0x99]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  99c009e3      <unknown>

ldapr x3, [x15], #8
// CHECK-INST: ldapr x3, [x15], #8
// CHECK-ENCODING: encoding: [0xe3,0x09,0xc0,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9c009e3      <unknown>

ldapr x3, [x15], 8
// CHECK-INST: ldapr x3, [x15], #8
// CHECK-ENCODING: encoding: [0xe3,0x09,0xc0,0xd9]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  d9c009e3      <unknown>

stlur b3, [x15, #-1]
// CHECK-INST: stlur b3, [x15, #-1]
// CHECK-ENCODING: encoding: [0xe3,0xf9,0x1f,0x1d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  1d1ff9e3      <unknown>

stlur h3, [x15, #2]
// CHECK-INST: stlur h3, [x15, #2]
// CHECK-ENCODING: encoding: [0xe3,0x29,0x00,0x5d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  5d0029e3      <unknown>

stlur s3, [x15, #-3]
// CHECK-INST: stlur s3, [x15, #-3]
// CHECK-ENCODING: encoding: [0xe3,0xd9,0x1f,0x9d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  9d1fd9e3      <unknown>

stlur d3, [sp, #4]
// CHECK-INST: stlur d3, [sp, #4]
// CHECK-ENCODING: encoding: [0xe3,0x4b,0x00,0xdd]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  dd004be3      <unknown>

stlur q3, [x15, #-5]
// CHECK-INST: stlur q3, [x15, #-5]
// CHECK-ENCODING: encoding: [0xe3,0xb9,0x9f,0x1d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  1d9fb9e3      <unknown>

ldapur b3, [x15, #6]
// CHECK-INST: ldapur b3, [x15, #6]
// CHECK-ENCODING: encoding: [0xe3,0x69,0x40,0x1d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  1d4069e3      <unknown>

ldapur h3, [x15, #-7]
// CHECK-INST: ldapur h3, [x15, #-7]
// CHECK-ENCODING: encoding: [0xe3,0x99,0x5f,0x5d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  5d5f99e3      <unknown>

ldapur s3, [x15, #8]
// CHECK-INST: ldapur s3, [x15, #8]
// CHECK-ENCODING: encoding: [0xe3,0x89,0x40,0x9d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  9d4089e3      <unknown>

ldapur d3, [x15, #-9]
// CHECK-INST: ldapur d3, [x15, #-9]
// CHECK-ENCODING: encoding: [0xe3,0x79,0x5f,0xdd]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  dd5f79e3      <unknown>

ldapur q3, [sp, #10]
// CHECK-INST: ldapur q3, [sp, #10]
// CHECK-ENCODING: encoding: [0xe3,0xab,0xc0,0x1d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  1dc0abe3      <unknown>

stl1  { v3.d }[0], [x15]
// CHECK-INST: stl1 { v3.d }[0], [x15]
// CHECK-ENCODING: encoding: [0xe3,0x85,0x01,0x0d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  0d0185e3      <unknown>

stl1  { v3.d }[0], [x15, #0]
// CHECK-INST: stl1 { v3.d }[0], [x15]
// CHECK-ENCODING: encoding: [0xe3,0x85,0x01,0x0d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  0d0185e3      <unknown>

stl1  { v3.d }[1], [sp]
// CHECK-INST: stl1 { v3.d }[1], [sp]
// CHECK-ENCODING: encoding: [0xe3,0x87,0x01,0x4d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  4d0187e3      <unknown>

stl1  { v3.d }[1], [sp, 0]
// CHECK-INST: stl1 { v3.d }[1], [sp]
// CHECK-ENCODING: encoding: [0xe3,0x87,0x01,0x4d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  4d0187e3      <unknown>

ldap1 { v3.d }[0], [sp]
// CHECK-INST: ldap1 { v3.d }[0], [sp]
// CHECK-ENCODING: encoding: [0xe3,0x87,0x41,0x0d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  0d4187e3      <unknown>

ldap1 { v3.d }[0], [sp, #0]
// CHECK-INST: ldap1 { v3.d }[0], [sp]
// CHECK-ENCODING: encoding: [0xe3,0x87,0x41,0x0d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  0d4187e3      <unknown>

ldap1 { v3.d }[1], [x15]
// CHECK-INST: ldap1 { v3.d }[1], [x15]
// CHECK-ENCODING: encoding: [0xe3,0x85,0x41,0x4d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  4d4185e3      <unknown>

ldap1 { v3.d }[1], [x15, 0]
// CHECK-INST: ldap1 { v3.d }[1], [x15]
// CHECK-ENCODING: encoding: [0xe3,0x85,0x41,0x4d]
// CHECK-ERROR:error: instruction requires: rcpc3
// CHECK-UNKNOWN:  4d4185e3      <unknown>
