// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

zip1    z0.b, z0.b, z0.b
// CHECK-INST: zip1    z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x60,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 20 05 <unknown>

zip1    z0.h, z0.h, z0.h
// CHECK-INST: zip1    z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x60,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 60 05 <unknown>

zip1    z0.s, z0.s, z0.s
// CHECK-INST: zip1    z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x60,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 a0 05 <unknown>

zip1    z0.d, z0.d, z0.d
// CHECK-INST: zip1    z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x60,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 e0 05 <unknown>

zip1    z31.b, z31.b, z31.b
// CHECK-INST: zip1    z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x63,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 63 3f 05 <unknown>

zip1    z31.h, z31.h, z31.h
// CHECK-INST: zip1    z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x63,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 63 7f 05 <unknown>

zip1    z31.s, z31.s, z31.s
// CHECK-INST: zip1    z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x63,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 63 bf 05 <unknown>

zip1    z31.d, z31.d, z31.d
// CHECK-INST: zip1    z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x63,0xff,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 63 ff 05 <unknown>

zip1    p0.b, p0.b, p0.b
// CHECK-INST: zip1    p0.b, p0.b, p0.b
// CHECK-ENCODING: [0x00,0x40,0x20,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 20 05 <unknown>

zip1    p0.h, p0.h, p0.h
// CHECK-INST: zip1    p0.h, p0.h, p0.h
// CHECK-ENCODING: [0x00,0x40,0x60,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 60 05 <unknown>

zip1    p0.s, p0.s, p0.s
// CHECK-INST: zip1    p0.s, p0.s, p0.s
// CHECK-ENCODING: [0x00,0x40,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 a0 05 <unknown>

zip1    p0.d, p0.d, p0.d
// CHECK-INST: zip1    p0.d, p0.d, p0.d
// CHECK-ENCODING: [0x00,0x40,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 e0 05 <unknown>

zip1    p15.b, p15.b, p15.b
// CHECK-INST: zip1    p15.b, p15.b, p15.b
// CHECK-ENCODING: [0xef,0x41,0x2f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 41 2f 05 <unknown>

zip1    p15.s, p15.s, p15.s
// CHECK-INST: zip1    p15.s, p15.s, p15.s
// CHECK-ENCODING: [0xef,0x41,0xaf,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 41 af 05 <unknown>

zip1    p15.h, p15.h, p15.h
// CHECK-INST: zip1    p15.h, p15.h, p15.h
// CHECK-ENCODING: [0xef,0x41,0x6f,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 41 6f 05 <unknown>

zip1    p15.d, p15.d, p15.d
// CHECK-INST: zip1    p15.d, p15.d, p15.d
// CHECK-ENCODING: [0xef,0x41,0xef,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef 41 ef 05 <unknown>
