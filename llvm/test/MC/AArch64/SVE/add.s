// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

add     z31.s, z31.s, z31.s
// CHECK-INST: add     z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x03,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 03 bf 04 <unknown>

add     z23.d, z13.d, z8.d
// CHECK-INST: add     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x01,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 01 e8 04 <unknown>

add     z0.s, z0.s, z0.s
// CHECK-INST: add     z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 a0 04 <unknown>

add     z31.d, z31.d, z31.d
// CHECK-INST: add     z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x03,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 03 ff 04 <unknown>

add     z21.b, z10.b, z21.b
// CHECK-INST: add     z21.b, z10.b, z21.b
// CHECK-ENCODING: [0x55,0x01,0x35,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 01 35 04 <unknown>

add     z31.b, z31.b, z31.b
// CHECK-INST: add     z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x03,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 03 3f 04 <unknown>

add     z0.h, z0.h, z0.h
// CHECK-INST: add     z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 60 04 <unknown>

add     z23.b, z13.b, z8.b
// CHECK-INST: add     z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x01,0x28,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 01 28 04 <unknown>

add     z0.d, z0.d, z0.d
// CHECK-INST: add     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 e0 04 <unknown>

add     z31.h, z31.h, z31.h
// CHECK-INST: add     z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x03,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 03 7f 04 <unknown>

add     z0.b, z0.b, z0.b
// CHECK-INST: add     z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 20 04 <unknown>

add     z21.d, z10.d, z21.d
// CHECK-INST: add     z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0x01,0xf5,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 01 f5 04 <unknown>

add     z21.h, z10.h, z21.h
// CHECK-INST: add     z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x01,0x75,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 01 75 04 <unknown>

add     z21.s, z10.s, z21.s
// CHECK-INST: add     z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0x01,0xb5,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 01 b5 04 <unknown>

add     z23.h, z13.h, z8.h
// CHECK-INST: add     z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x01,0x68,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 01 68 04 <unknown>

add     z23.s, z13.s, z8.s
// CHECK-INST: add     z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0x01,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 01 a8 04 <unknown>
