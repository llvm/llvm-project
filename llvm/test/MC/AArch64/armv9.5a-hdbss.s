// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



mrs x0, HDBSSBR_EL2
// CHECK-INST: mrs x0, HDBSSBR_EL2
// CHECK-ENCODING: encoding: [0x40,0x23,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c2340 mrs x0, HDBSSBR_EL2

msr HDBSSBR_EL2, x0
// CHECK-INST: msr HDBSSBR_EL2, x0
// CHECK-ENCODING: encoding: [0x40,0x23,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2340 msr HDBSSBR_EL2, x0

mrs x0, HDBSSPROD_EL2
// CHECK-INST: mrs x0, HDBSSPROD_EL2
// CHECK-ENCODING: encoding: [0x60,0x23,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c2360 mrs x0, HDBSSPROD_EL2

msr HDBSSPROD_EL2, x0
// CHECK-INST: msr HDBSSPROD_EL2, x0
// CHECK-ENCODING: encoding: [0x60,0x23,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2360 msr HDBSSPROD_EL2, x0

