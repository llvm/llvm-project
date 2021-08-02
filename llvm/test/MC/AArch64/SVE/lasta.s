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

lasta   w0, p7, z31.b
// CHECK-INST: lasta	w0, p7, z31.b
// CHECK-ENCODING: [0xe0,0xbf,0x20,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bf 20 05 <unknown>

lasta   w0, p7, z31.h
// CHECK-INST: lasta	w0, p7, z31.h
// CHECK-ENCODING: [0xe0,0xbf,0x60,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bf 60 05 <unknown>

lasta   w0, p7, z31.s
// CHECK-INST: lasta	w0, p7, z31.s
// CHECK-ENCODING: [0xe0,0xbf,0xa0,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bf a0 05 <unknown>

lasta   x0, p7, z31.d
// CHECK-INST: lasta	x0, p7, z31.d
// CHECK-ENCODING: [0xe0,0xbf,0xe0,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bf e0 05 <unknown>

lasta   b0, p7, z31.b
// CHECK-INST: lasta	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x22,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f 22 05 <unknown>

lasta   h0, p7, z31.h
// CHECK-INST: lasta	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x62,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f 62 05 <unknown>

lasta   s0, p7, z31.s
// CHECK-INST: lasta	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xa2,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f a2 05 <unknown>

lasta   d0, p7, z31.d
// CHECK-INST: lasta	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe2,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 9f e2 05 <unknown>
