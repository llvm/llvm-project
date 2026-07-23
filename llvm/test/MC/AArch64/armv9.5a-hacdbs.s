// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


mrs x0, HACDBSBR_EL2
// CHECK-INST: mrs x0, HACDBSBR_EL2
// CHECK-ENCODING: encoding: [0x80,0x23,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c2380 mrs x0, HACDBSBR_EL2

msr HACDBSBR_EL2, x0
// CHECK-INST: msr HACDBSBR_EL2, x0
// CHECK-ENCODING: encoding: [0x80,0x23,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2380 msr HACDBSBR_EL2, x0

mrs x0, HACDBSCONS_EL2
// CHECK-INST: mrs x0, HACDBSCONS_EL2
// CHECK-ENCODING: encoding: [0xa0,0x23,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c23a0 mrs x0, HACDBSCONS_EL2

msr HACDBSCONS_EL2, x0
// CHECK-INST: msr HACDBSCONS_EL2, x0
// CHECK-ENCODING: encoding: [0xa0,0x23,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c23a0 msr HACDBSCONS_EL2, x0

