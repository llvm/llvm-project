// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

srls
// CHECK-INST: srls
// CHECK-ENCODING: encoding: [0xdf,0x26,0x03,0xd5]

srls stshstrm
// CHECK-INST: srls stshstrm
// CHECK-ENCODING: encoding: [0xff,0x26,0x03,0xd5]

slbnd
// CHECK-INST: slbnd
// CHECK-ENCODING: encoding: [0x1f,0x27,0x03,0xd5]

hint #54
// CHECK-INST: srls
// CHECK-ENCODING: encoding: [0xdf,0x26,0x03,0xd5]

hint #55
// CHECK-INST: srls stshstrm
// CHECK-ENCODING: encoding: [0xff,0x26,0x03,0xd5]

hint #56
// CHECK-INST: slbnd
// CHECK-ENCODING: encoding: [0x1f,0x27,0x03,0xd5]
