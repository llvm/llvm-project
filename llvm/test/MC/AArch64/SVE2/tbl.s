// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

tbl  z28.b, { z29.b, z30.b }, z31.b
// CHECK-INST: tbl  z28.b, { z29.b, z30.b }, z31.b
// CHECK-ENCODING: [0xbc,0x2b,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: bc 2b 3f 05 <unknown>

tbl  z28.h, { z29.h, z30.h }, z31.h
// CHECK-INST: tbl  z28.h, { z29.h, z30.h }, z31.h
// CHECK-ENCODING: [0xbc,0x2b,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: bc 2b 7f 05 <unknown>

tbl  z28.s, { z29.s, z30.s }, z31.s
// CHECK-INST: tbl  z28.s, { z29.s, z30.s }, z31.s
// CHECK-ENCODING: [0xbc,0x2b,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: bc 2b bf 05 <unknown>

tbl  z28.d, { z29.d, z30.d }, z31.d
// CHECK-INST: tbl  z28.d, { z29.d, z30.d }, z31.d
// CHECK-ENCODING: [0xbc,0x2b,0xff,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: bc 2b ff 05 <unknown>
