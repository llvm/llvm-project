// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

incb    x0
// CHECK-INST: incb    x0
// CHECK-ENCODING: [0xe0,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e3 30 04 <unknown>

incb    x0, all
// CHECK-INST: incb    x0
// CHECK-ENCODING: [0xe0,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e3 30 04 <unknown>

incb    x0, all, mul #1
// CHECK-INST: incb    x0
// CHECK-ENCODING: [0xe0,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e3 30 04 <unknown>

incb    x0, all, mul #16
// CHECK-INST: incb    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe3,0x3f,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e3 3f 04 <unknown>

incb    x0, pow2
// CHECK-INST: incb    x0, pow2
// CHECK-ENCODING: [0x00,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e0 30 04 <unknown>

incb    x0, vl1
// CHECK-INST: incb    x0, vl1
// CHECK-ENCODING: [0x20,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 e0 30 04 <unknown>

incb    x0, vl2
// CHECK-INST: incb    x0, vl2
// CHECK-ENCODING: [0x40,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 e0 30 04 <unknown>

incb    x0, vl3
// CHECK-INST: incb    x0, vl3
// CHECK-ENCODING: [0x60,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 e0 30 04 <unknown>

incb    x0, vl4
// CHECK-INST: incb    x0, vl4
// CHECK-ENCODING: [0x80,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 e0 30 04 <unknown>

incb    x0, vl5
// CHECK-INST: incb    x0, vl5
// CHECK-ENCODING: [0xa0,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 e0 30 04 <unknown>

incb    x0, vl6
// CHECK-INST: incb    x0, vl6
// CHECK-ENCODING: [0xc0,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 e0 30 04 <unknown>

incb    x0, vl7
// CHECK-INST: incb    x0, vl7
// CHECK-ENCODING: [0xe0,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e0 30 04 <unknown>

incb    x0, vl8
// CHECK-INST: incb    x0, vl8
// CHECK-ENCODING: [0x00,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e1 30 04 <unknown>

incb    x0, vl16
// CHECK-INST: incb    x0, vl16
// CHECK-ENCODING: [0x20,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 e1 30 04 <unknown>

incb    x0, vl32
// CHECK-INST: incb    x0, vl32
// CHECK-ENCODING: [0x40,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 e1 30 04 <unknown>

incb    x0, vl64
// CHECK-INST: incb    x0, vl64
// CHECK-ENCODING: [0x60,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 e1 30 04 <unknown>

incb    x0, vl128
// CHECK-INST: incb    x0, vl128
// CHECK-ENCODING: [0x80,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 e1 30 04 <unknown>

incb    x0, vl256
// CHECK-INST: incb    x0, vl256
// CHECK-ENCODING: [0xa0,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 e1 30 04 <unknown>

incb    x0, #14
// CHECK-INST: incb    x0, #14
// CHECK-ENCODING: [0xc0,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 e1 30 04 <unknown>

incb    x0, #15
// CHECK-INST: incb    x0, #15
// CHECK-ENCODING: [0xe0,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e1 30 04 <unknown>

incb    x0, #16
// CHECK-INST: incb    x0, #16
// CHECK-ENCODING: [0x00,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e2 30 04 <unknown>

incb    x0, #17
// CHECK-INST: incb    x0, #17
// CHECK-ENCODING: [0x20,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 e2 30 04 <unknown>

incb    x0, #18
// CHECK-INST: incb    x0, #18
// CHECK-ENCODING: [0x40,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 e2 30 04 <unknown>

incb    x0, #19
// CHECK-INST: incb    x0, #19
// CHECK-ENCODING: [0x60,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 e2 30 04 <unknown>

incb    x0, #20
// CHECK-INST: incb    x0, #20
// CHECK-ENCODING: [0x80,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 e2 30 04 <unknown>

incb    x0, #21
// CHECK-INST: incb    x0, #21
// CHECK-ENCODING: [0xa0,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 e2 30 04 <unknown>

incb    x0, #22
// CHECK-INST: incb    x0, #22
// CHECK-ENCODING: [0xc0,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 e2 30 04 <unknown>

incb    x0, #23
// CHECK-INST: incb    x0, #23
// CHECK-ENCODING: [0xe0,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 e2 30 04 <unknown>

incb    x0, #24
// CHECK-INST: incb    x0, #24
// CHECK-ENCODING: [0x00,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e3 30 04 <unknown>

incb    x0, #25
// CHECK-INST: incb    x0, #25
// CHECK-ENCODING: [0x20,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 e3 30 04 <unknown>

incb    x0, #26
// CHECK-INST: incb    x0, #26
// CHECK-ENCODING: [0x40,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 e3 30 04 <unknown>

incb    x0, #27
// CHECK-INST: incb    x0, #27
// CHECK-ENCODING: [0x60,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 e3 30 04 <unknown>

incb    x0, #28
// CHECK-INST: incb    x0, #28
// CHECK-ENCODING: [0x80,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 e3 30 04 <unknown>
