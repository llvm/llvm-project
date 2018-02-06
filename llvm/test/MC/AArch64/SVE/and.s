// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

and     z5.b, z5.b, #0xf9
// CHECK-INST: and     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x80,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5 2e 80 05 <unknown>

bic     z5.b, z5.b, #0x06
// CHECK-INST: and     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x80,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5 2e 80 05 <unknown>

and     z23.h, z23.h, #0xfff9
// CHECK-INST: and     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x80,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 6d 80 05 <unknown>

bic     z23.h, z23.h, #0x0006
// CHECK-INST: and     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x80,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 6d 80 05 <unknown>

and     z0.s, z0.s, #0xfffffff9
// CHECK-INST: and     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x80,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 eb 80 05 <unknown>

bic     z0.s, z0.s, #0x00000006
// CHECK-INST: and     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x80,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 eb 80 05 <unknown>

and     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: and     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x83,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 ef 83 05 <unknown>

bic     z0.d, z0.d, #0x0000000000000006
// CHECK-INST: and     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x83,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a0 ef 83 05 <unknown>
