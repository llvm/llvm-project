// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



msr pmbmar_el1, x0
// CHECK-INST: msr PMBMAR_EL1, x0
// CHECK-ENCODING: encoding: [0xa0,0x9a,0x18,0xd5]
// CHECK-UNKNOWN:  d5189aa0 msr PMBMAR_EL1, x0

msr pmbsr_el12, x0
// CHECK-INST: msr PMBSR_EL12, x0
// CHECK-ENCODING: encoding: [0x60,0x9a,0x1d,0xd5]
// CHECK-UNKNOWN:  d51d9a60 msr PMBSR_EL12, x0

msr pmbsr_el2, x0
// CHECK-INST: msr PMBSR_EL2, x0
// CHECK-ENCODING: encoding: [0x60,0x9a,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c9a60 msr PMBSR_EL2, x0

msr pmbsr_el3, x0
// CHECK-INST: msr PMBSR_EL3, x0
// CHECK-ENCODING: encoding: [0x60,0x9a,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e9a60 msr PMBSR_EL3, x0

mrs x0, pmbmar_el1
// CHECK-INST: mrs x0, PMBMAR_EL1
// CHECK-ENCODING: encoding: [0xa0,0x9a,0x38,0xd5]
// CHECK-UNKNOWN:  d5389aa0 mrs x0, PMBMAR_EL1

mrs x0, pmbsr_el12
// CHECK-INST: mrs x0, PMBSR_EL12
// CHECK-ENCODING: encoding: [0x60,0x9a,0x3d,0xd5]
// CHECK-UNKNOWN:  d53d9a60 mrs x0, PMBSR_EL12

mrs x0, pmbsr_el2
// CHECK-INST: mrs x0, PMBSR_EL2
// CHECK-ENCODING: encoding: [0x60,0x9a,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c9a60 mrs x0, PMBSR_EL2

mrs x0, pmbsr_el3
// CHECK-INST: mrs x0, PMBSR_EL3
// CHECK-ENCODING: encoding: [0x60,0x9a,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e9a60 mrs x0, PMBSR_EL3
