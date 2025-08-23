// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


mrs x0, FGWTE3_EL3
// CHECK-INST: mrs x0, FGWTE3_EL3
// CHECK-ENCODING: encoding: [0xa0,0x11,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e11a0 mrs x0, FGWTE3_EL3

msr FGWTE3_EL3, x0
// CHECK-INST: msr FGWTE3_EL3, x0
// CHECK-ENCODING: encoding: [0xa0,0x11,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e11a0 msr FGWTE3_EL3, x0
