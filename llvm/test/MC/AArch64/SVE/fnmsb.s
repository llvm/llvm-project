// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fnmsb z0.h, p7/m, z1.h, z31.h
// CHECK-INST: fnmsb	z0.h, p7/m, z1.h, z31.h
// CHECK-ENCODING: [0x20,0xfc,0x7f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 fc 7f 65 <unknown>

fnmsb z0.s, p7/m, z1.s, z31.s
// CHECK-INST: fnmsb	z0.s, p7/m, z1.s, z31.s
// CHECK-ENCODING: [0x20,0xfc,0xbf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 fc bf 65 <unknown>

fnmsb z0.d, p7/m, z1.d, z31.d
// CHECK-INST: fnmsb	z0.d, p7/m, z1.d, z31.d
// CHECK-ENCODING: [0x20,0xfc,0xff,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 fc ff 65 <unknown>
