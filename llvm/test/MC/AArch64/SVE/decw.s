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

decw    x0
// CHECK-INST: decw    x0
// CHECK-ENCODING: [0xe0,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 e7 b0 04 <unknown>

decw    x0, all
// CHECK-INST: decw    x0
// CHECK-ENCODING: [0xe0,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 e7 b0 04 <unknown>

decw    x0, all, mul #1
// CHECK-INST: decw    x0
// CHECK-ENCODING: [0xe0,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 e7 b0 04 <unknown>

decw    x0, all, mul #16
// CHECK-INST: decw    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe7,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 e7 bf 04 <unknown>

decw    x0, pow2
// CHECK-INST: decw    x0, pow2
// CHECK-ENCODING: [0x00,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e4 b0 04 <unknown>

decw    x0, vl1
// CHECK-INST: decw    x0, vl1
// CHECK-ENCODING: [0x20,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 e4 b0 04 <unknown>

decw    x0, vl2
// CHECK-INST: decw    x0, vl2
// CHECK-ENCODING: [0x40,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 e4 b0 04 <unknown>

decw    x0, vl3
// CHECK-INST: decw    x0, vl3
// CHECK-ENCODING: [0x60,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 e4 b0 04 <unknown>

decw    x0, vl4
// CHECK-INST: decw    x0, vl4
// CHECK-ENCODING: [0x80,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 e4 b0 04 <unknown>

decw    x0, vl5
// CHECK-INST: decw    x0, vl5
// CHECK-ENCODING: [0xa0,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 e4 b0 04 <unknown>

decw    x0, vl6
// CHECK-INST: decw    x0, vl6
// CHECK-ENCODING: [0xc0,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 e4 b0 04 <unknown>

decw    x0, vl7
// CHECK-INST: decw    x0, vl7
// CHECK-ENCODING: [0xe0,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 e4 b0 04 <unknown>

decw    x0, vl8
// CHECK-INST: decw    x0, vl8
// CHECK-ENCODING: [0x00,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e5 b0 04 <unknown>

decw    x0, vl16
// CHECK-INST: decw    x0, vl16
// CHECK-ENCODING: [0x20,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 e5 b0 04 <unknown>

decw    x0, vl32
// CHECK-INST: decw    x0, vl32
// CHECK-ENCODING: [0x40,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 e5 b0 04 <unknown>

decw    x0, vl64
// CHECK-INST: decw    x0, vl64
// CHECK-ENCODING: [0x60,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 e5 b0 04 <unknown>

decw    x0, vl128
// CHECK-INST: decw    x0, vl128
// CHECK-ENCODING: [0x80,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 e5 b0 04 <unknown>

decw    x0, vl256
// CHECK-INST: decw    x0, vl256
// CHECK-ENCODING: [0xa0,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 e5 b0 04 <unknown>

decw    x0, #14
// CHECK-INST: decw    x0, #14
// CHECK-ENCODING: [0xc0,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 e5 b0 04 <unknown>

decw    x0, #28
// CHECK-INST: decw    x0, #28
// CHECK-ENCODING: [0x80,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 e7 b0 04 <unknown>
