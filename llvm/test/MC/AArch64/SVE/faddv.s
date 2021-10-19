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

faddv h0, p7, z31.h
// CHECK-INST: faddv	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x3f,0x40,0x65]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 3f 40 65 <unknown>

faddv s0, p7, z31.s
// CHECK-INST: faddv	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x3f,0x80,0x65]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 3f 80 65 <unknown>

faddv d0, p7, z31.d
// CHECK-INST: faddv	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x3f,0xc0,0x65]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 3f c0 65 <unknown>
