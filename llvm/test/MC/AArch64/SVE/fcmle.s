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

fcmle   p0.h, p0/z, z0.h, #0.0
// CHECK-INST: fcmle	p0.h, p0/z, z0.h, #0.0
// CHECK-ENCODING: [0x10,0x20,0x51,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 10 20 51 65 <unknown>

fcmle   p0.s, p0/z, z0.s, #0.0
// CHECK-INST: fcmle	p0.s, p0/z, z0.s, #0.0
// CHECK-ENCODING: [0x10,0x20,0x91,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 10 20 91 65 <unknown>

fcmle   p0.d, p0/z, z0.d, #0.0
// CHECK-INST: fcmle	p0.d, p0/z, z0.d, #0.0
// CHECK-ENCODING: [0x10,0x20,0xd1,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 10 20 d1 65 <unknown>

fcmle   p0.h, p0/z, z0.h, z1.h
// CHECK-INST: fcmge	p0.h, p0/z, z1.h, z0.h
// CHECK-ENCODING: [0x20,0x40,0x40,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 40 40 65 <unknown>

fcmle   p0.s, p0/z, z0.s, z1.s
// CHECK-INST: fcmge	p0.s, p0/z, z1.s, z0.s
// CHECK-ENCODING: [0x20,0x40,0x80,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 40 80 65 <unknown>

fcmle   p0.d, p0/z, z0.d, z1.d
// CHECK-INST: fcmge	p0.d, p0/z, z1.d, z0.d
// CHECK-ENCODING: [0x20,0x40,0xc0,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 40 c0 65 <unknown>
