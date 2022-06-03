// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

tbl  z31.b, z31.b, z31.b
// CHECK-INST: tbl	z31.b, { z31.b }, z31.b
// CHECK-ENCODING: [0xff,0x33,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 3f 05 <unknown>

tbl  z31.h, z31.h, z31.h
// CHECK-INST: tbl	z31.h, { z31.h }, z31.h
// CHECK-ENCODING: [0xff,0x33,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 7f 05 <unknown>

tbl  z31.s, z31.s, z31.s
// CHECK-INST: tbl	z31.s, { z31.s }, z31.s
// CHECK-ENCODING: [0xff,0x33,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 bf 05 <unknown>

tbl  z31.d, z31.d, z31.d
// CHECK-INST: tbl	z31.d, { z31.d }, z31.d
// CHECK-ENCODING: [0xff,0x33,0xff,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 ff 05 <unknown>

tbl  z31.b, { z31.b }, z31.b
// CHECK-INST: tbl	z31.b, { z31.b }, z31.b
// CHECK-ENCODING: [0xff,0x33,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 3f 05 <unknown>

tbl  z31.h, { z31.h }, z31.h
// CHECK-INST: tbl	z31.h, { z31.h }, z31.h
// CHECK-ENCODING: [0xff,0x33,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 7f 05 <unknown>

tbl  z31.s, { z31.s }, z31.s
// CHECK-INST: tbl	z31.s, { z31.s }, z31.s
// CHECK-ENCODING: [0xff,0x33,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 bf 05 <unknown>

tbl  z31.d, { z31.d }, z31.d
// CHECK-INST: tbl	z31.d, { z31.d }, z31.d
// CHECK-ENCODING: [0xff,0x33,0xff,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 33 ff 05 <unknown>
