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

ctermne w30, wzr
// CHECK-INST: ctermne	w30, wzr
// CHECK-ENCODING: [0xd0,0x23,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: d0 23 bf 25 <unknown>

ctermne wzr, w30
// CHECK-INST: ctermne	wzr, w30
// CHECK-ENCODING: [0xf0,0x23,0xbe,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f0 23 be 25 <unknown>

ctermne x30, xzr
// CHECK-INST: ctermne	x30, xzr
// CHECK-ENCODING: [0xd0,0x23,0xff,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: d0 23 ff 25 <unknown>

ctermne xzr, x30
// CHECK-INST: ctermne	xzr, x30
// CHECK-ENCODING: [0xf0,0x23,0xfe,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f0 23 fe 25 <unknown>
