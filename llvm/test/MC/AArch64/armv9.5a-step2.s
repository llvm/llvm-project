// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



mrs x0, MDSTEPOP_EL1
// CHECK-INST: mrs x0, MDSTEPOP_EL1
// CHECK-ENCODING: encoding: [0x40,0x05,0x30,0xd5]
// CHECK-UNKNOWN:  d5300540 mrs x0, MDSTEPOP_EL1

msr MDSTEPOP_EL1, x0
// CHECK-INST: msr MDSTEPOP_EL1, x0
// CHECK-ENCODING: encoding: [0x40,0x05,0x10,0xd5]
// CHECK-UNKNOWN:  d5100540 msr MDSTEPOP_EL1, x0
