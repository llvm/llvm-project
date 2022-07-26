// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

decw    x0
// CHECK-INST: decw    x0
// CHECK-ENCODING: [0xe0,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e7e0 <unknown>

decw    x0, all
// CHECK-INST: decw    x0
// CHECK-ENCODING: [0xe0,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e7e0 <unknown>

decw    x0, all, mul #1
// CHECK-INST: decw    x0
// CHECK-ENCODING: [0xe0,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e7e0 <unknown>

decw    x0, all, mul #16
// CHECK-INST: decw    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe7,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bfe7e0 <unknown>

decw    x0, pow2
// CHECK-INST: decw    x0, pow2
// CHECK-ENCODING: [0x00,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e400 <unknown>

decw    x0, vl1
// CHECK-INST: decw    x0, vl1
// CHECK-ENCODING: [0x20,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e420 <unknown>

decw    x0, vl2
// CHECK-INST: decw    x0, vl2
// CHECK-ENCODING: [0x40,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e440 <unknown>

decw    x0, vl3
// CHECK-INST: decw    x0, vl3
// CHECK-ENCODING: [0x60,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e460 <unknown>

decw    x0, vl4
// CHECK-INST: decw    x0, vl4
// CHECK-ENCODING: [0x80,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e480 <unknown>

decw    x0, vl5
// CHECK-INST: decw    x0, vl5
// CHECK-ENCODING: [0xa0,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e4a0 <unknown>

decw    x0, vl6
// CHECK-INST: decw    x0, vl6
// CHECK-ENCODING: [0xc0,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e4c0 <unknown>

decw    x0, vl7
// CHECK-INST: decw    x0, vl7
// CHECK-ENCODING: [0xe0,0xe4,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e4e0 <unknown>

decw    x0, vl8
// CHECK-INST: decw    x0, vl8
// CHECK-ENCODING: [0x00,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e500 <unknown>

decw    x0, vl16
// CHECK-INST: decw    x0, vl16
// CHECK-ENCODING: [0x20,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e520 <unknown>

decw    x0, vl32
// CHECK-INST: decw    x0, vl32
// CHECK-ENCODING: [0x40,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e540 <unknown>

decw    x0, vl64
// CHECK-INST: decw    x0, vl64
// CHECK-ENCODING: [0x60,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e560 <unknown>

decw    x0, vl128
// CHECK-INST: decw    x0, vl128
// CHECK-ENCODING: [0x80,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e580 <unknown>

decw    x0, vl256
// CHECK-INST: decw    x0, vl256
// CHECK-ENCODING: [0xa0,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e5a0 <unknown>

decw    x0, #14
// CHECK-INST: decw    x0, #14
// CHECK-ENCODING: [0xc0,0xe5,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e5c0 <unknown>

decw    x0, #28
// CHECK-INST: decw    x0, #28
// CHECK-ENCODING: [0x80,0xe7,0xb0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b0e780 <unknown>
