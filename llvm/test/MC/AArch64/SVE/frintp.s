// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

frintp   z31.h, p7/m, z31.h
// CHECK-INST: frintp	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x41,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 41 65 <unknown>

frintp   z31.s, p7/m, z31.s
// CHECK-INST: frintp	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x81,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 81 65 <unknown>

frintp   z31.d, p7/m, z31.d
// CHECK-INST: frintp	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc1,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf c1 65 <unknown>
