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

st1b    z0.b, p0, [x0]
// CHECK-INST: st1b    { z0.b }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x00,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e400e000 <unknown>

st1b    z0.h, p0, [x0]
// CHECK-INST: st1b    { z0.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x20,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e420e000 <unknown>

st1b    z0.s, p0, [x0]
// CHECK-INST: st1b    { z0.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e440e000 <unknown>

st1b    z0.d, p0, [x0]
// CHECK-INST: st1b    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e460e000 <unknown>

st1b    { z0.b }, p0, [x0]
// CHECK-INST: st1b    { z0.b }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x00,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e400e000 <unknown>

st1b    { z0.h }, p0, [x0]
// CHECK-INST: st1b    { z0.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x20,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e420e000 <unknown>

st1b    { z0.s }, p0, [x0]
// CHECK-INST: st1b    { z0.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e440e000 <unknown>

st1b    { z0.d }, p0, [x0]
// CHECK-INST: st1b    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e460e000 <unknown>

st1b    { z31.b }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1b    { z31.b }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x0f,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e40fffff <unknown>

st1b    { z21.b }, p5, [x10, #5, mul vl]
// CHECK-INST: st1b    { z21.b }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x05,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e405f555 <unknown>

st1b    { z31.h }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1b    { z31.h }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x2f,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e42fffff <unknown>

st1b    { z21.h }, p5, [x10, #5, mul vl]
// CHECK-INST: st1b    { z21.h }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x25,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e425f555 <unknown>

st1b    { z31.s }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1b    { z31.s }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x4f,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e44fffff <unknown>

st1b    { z21.s }, p5, [x10, #5, mul vl]
// CHECK-INST: st1b    { z21.s }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x45,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e445f555 <unknown>

st1b    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1b    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x6f,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e46fffff <unknown>

st1b    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-INST: st1b    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x65,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e465f555 <unknown>

st1b    { z0.b }, p0, [x0, x0]
// CHECK-INST: st1b    { z0.b }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0x00,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4004000 <unknown>

st1b    { z0.h }, p0, [x0, x0]
// CHECK-INST: st1b    { z0.h }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0x20,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4204000 <unknown>

st1b    { z0.s }, p0, [x0, x0]
// CHECK-INST: st1b    { z0.s }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4404000 <unknown>

st1b    { z0.d }, p0, [x0, x0]
// CHECK-INST: st1b    { z0.d }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0x60,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4604000 <unknown>
