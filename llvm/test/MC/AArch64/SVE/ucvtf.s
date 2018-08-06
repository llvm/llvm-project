// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ucvtf   z0.h, p0/m, z0.h
// CHECK-INST: ucvtf   z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x53,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 53 65 <unknown>

ucvtf   z0.h, p0/m, z0.s
// CHECK-INST: ucvtf   z0.h, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x55,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 55 65 <unknown>

ucvtf   z0.h, p0/m, z0.d
// CHECK-INST: ucvtf   z0.h, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0x57,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 57 65 <unknown>

ucvtf   z0.s, p0/m, z0.s
// CHECK-INST: ucvtf   z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x95,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 95 65 <unknown>

ucvtf   z0.s, p0/m, z0.d
// CHECK-INST: ucvtf   z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd5,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d5 65 <unknown>

ucvtf   z0.d, p0/m, z0.s
// CHECK-INST: ucvtf   z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xd1,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d1 65 <unknown>

ucvtf   z0.d, p0/m, z0.d
// CHECK-INST: ucvtf   z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d7 65 <unknown>
