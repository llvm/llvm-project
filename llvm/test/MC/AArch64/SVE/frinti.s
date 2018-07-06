// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

frinti   z31.h, p7/m, z31.h
// CHECK-INST: frinti	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x47,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 47 65 <unknown>

frinti   z31.s, p7/m, z31.s
// CHECK-INST: frinti	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x87,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 87 65 <unknown>

frinti   z31.d, p7/m, z31.d
// CHECK-INST: frinti	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc7,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf c7 65 <unknown>
