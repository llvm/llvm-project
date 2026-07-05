// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

decd    x0
// CHECK-INST: decd    x0
// CHECK-ENCODING: [0xe0,0xe7,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e7e0 <unknown>

decd    x0, all
// CHECK-INST: decd    x0
// CHECK-ENCODING: [0xe0,0xe7,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e7e0 <unknown>

decd    x0, all, mul #1
// CHECK-INST: decd    x0
// CHECK-ENCODING: [0xe0,0xe7,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e7e0 <unknown>

decd    x0, all, mul #16
// CHECK-INST: decd    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe7,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ffe7e0 <unknown>

decd    x0, pow2
// CHECK-INST: decd    x0, pow2
// CHECK-ENCODING: [0x00,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e400 <unknown>

decd    x0, vl1
// CHECK-INST: decd    x0, vl1
// CHECK-ENCODING: [0x20,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e420 <unknown>

decd    x0, vl2
// CHECK-INST: decd    x0, vl2
// CHECK-ENCODING: [0x40,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e440 <unknown>

decd    x0, vl3
// CHECK-INST: decd    x0, vl3
// CHECK-ENCODING: [0x60,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e460 <unknown>

decd    x0, vl4
// CHECK-INST: decd    x0, vl4
// CHECK-ENCODING: [0x80,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e480 <unknown>

decd    x0, vl5
// CHECK-INST: decd    x0, vl5
// CHECK-ENCODING: [0xa0,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e4a0 <unknown>

decd    x0, vl6
// CHECK-INST: decd    x0, vl6
// CHECK-ENCODING: [0xc0,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e4c0 <unknown>

decd    x0, vl7
// CHECK-INST: decd    x0, vl7
// CHECK-ENCODING: [0xe0,0xe4,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e4e0 <unknown>

decd    x0, vl8
// CHECK-INST: decd    x0, vl8
// CHECK-ENCODING: [0x00,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e500 <unknown>

decd    x0, vl16
// CHECK-INST: decd    x0, vl16
// CHECK-ENCODING: [0x20,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e520 <unknown>

decd    x0, vl32
// CHECK-INST: decd    x0, vl32
// CHECK-ENCODING: [0x40,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e540 <unknown>

decd    x0, vl64
// CHECK-INST: decd    x0, vl64
// CHECK-ENCODING: [0x60,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e560 <unknown>

decd    x0, vl128
// CHECK-INST: decd    x0, vl128
// CHECK-ENCODING: [0x80,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e580 <unknown>

decd    x0, vl256
// CHECK-INST: decd    x0, vl256
// CHECK-ENCODING: [0xa0,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e5a0 <unknown>

decd    x0, #14
// CHECK-INST: decd    x0, #14
// CHECK-ENCODING: [0xc0,0xe5,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e5c0 <unknown>

decd    x0, #28
// CHECK-INST: decd    x0, #28
// CHECK-ENCODING: [0x80,0xe7,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0e780 <unknown>
