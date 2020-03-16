// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

zip2    z0.b, z0.b, z0.b
// CHECK-INST: zip2    z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x64,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 64 20 05 <unknown>

zip2    z0.h, z0.h, z0.h
// CHECK-INST: zip2    z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x64,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 64 60 05 <unknown>

zip2    z0.s, z0.s, z0.s
// CHECK-INST: zip2    z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x64,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 64 a0 05 <unknown>

zip2    z0.d, z0.d, z0.d
// CHECK-INST: zip2    z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x64,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 64 e0 05 <unknown>

zip2    z31.b, z31.b, z31.b
// CHECK-INST: zip2    z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x67,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 67 3f 05 <unknown>

zip2    z31.h, z31.h, z31.h
// CHECK-INST: zip2    z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x67,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 67 7f 05 <unknown>

zip2    z31.s, z31.s, z31.s
// CHECK-INST: zip2    z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x67,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 67 bf 05 <unknown>

zip2    z31.d, z31.d, z31.d
// CHECK-INST: zip2    z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x67,0xff,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 67 ff 05 <unknown>

zip2    p0.b, p0.b, p0.b
// CHECK-INST: zip2    p0.b, p0.b, p0.b
// CHECK-ENCODING: [0x00,0x44,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 44 20 05 <unknown>

zip2    p0.h, p0.h, p0.h
// CHECK-INST: zip2    p0.h, p0.h, p0.h
// CHECK-ENCODING: [0x00,0x44,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 44 60 05 <unknown>

zip2    p0.s, p0.s, p0.s
// CHECK-INST: zip2    p0.s, p0.s, p0.s
// CHECK-ENCODING: [0x00,0x44,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 44 a0 05 <unknown>

zip2    p0.d, p0.d, p0.d
// CHECK-INST: zip2    p0.d, p0.d, p0.d
// CHECK-ENCODING: [0x00,0x44,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 44 e0 05 <unknown>

zip2    p15.b, p15.b, p15.b
// CHECK-INST: zip2    p15.b, p15.b, p15.b
// CHECK-ENCODING: [0xef,0x45,0x2f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 45 2f 05 <unknown>

zip2    p15.h, p15.h, p15.h
// CHECK-INST: zip2    p15.h, p15.h, p15.h
// CHECK-ENCODING: [0xef,0x45,0x6f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 45 6f 05 <unknown>

zip2    p15.s, p15.s, p15.s
// CHECK-INST: zip2    p15.s, p15.s, p15.s
// CHECK-ENCODING: [0xef,0x45,0xaf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 45 af 05 <unknown>

zip2    p15.d, p15.d, p15.d
// CHECK-INST: zip2    p15.d, p15.d, p15.d
// CHECK-ENCODING: [0xef,0x45,0xef,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 45 ef 05 <unknown>
