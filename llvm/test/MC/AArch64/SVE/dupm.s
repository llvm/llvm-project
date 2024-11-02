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

dupm     z5.b, #0xf9
// CHECK-INST: dupm     z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5 2e c0 05 <unknown>

dupm     z5.h, #0xf9f9
// CHECK-INST: dupm     z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5 2e c0 05 <unknown>

dupm     z5.s, #0xf9f9f9f9
// CHECK-INST: dupm     z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5 2e c0 05 <unknown>

dupm     z5.d, #0xf9f9f9f9f9f9f9f9
// CHECK-INST: dupm     z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5 2e c0 05 <unknown>

dupm     z23.h, #0xfff9
// CHECK-INST: dupm     z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 6d c0 05 <unknown>

dupm     z23.s, #0xfff9fff9
// CHECK-INST: dupm     z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 6d c0 05 <unknown>

dupm     z23.d, #0xfff9fff9fff9fff9
// CHECK-INST: dupm     z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 6d c0 05 <unknown>

dupm     z0.s, #0xfffffff9
// CHECK-INST: dupm     z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 eb c0 05 <unknown>

dupm     z0.d, #0xfffffff9fffffff9
// CHECK-INST: dupm     z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0xc0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 eb c0 05 <unknown>

dupm     z0.d, #0xfffffffffffffff9
// CHECK-INST: dupm     z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0xc3,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 ef c3 05 <unknown>
