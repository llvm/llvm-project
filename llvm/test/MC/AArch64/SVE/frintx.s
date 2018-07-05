// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

frintx   z31.h, p7/m, z31.h
// CHECK-INST: frintx	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x46,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 46 65 <unknown>

frintx   z31.s, p7/m, z31.s
// CHECK-INST: frintx	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x86,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 86 65 <unknown>

frintx   z31.d, p7/m, z31.d
// CHECK-INST: frintx	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xc6,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf c6 65 <unknown>
