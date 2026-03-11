// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

stshh keep
// CHECK-INST: stshh keep
// CHECK-ENCODING: encoding: [0x1f,0x96,0x01,0xd5]

stshh strm
// CHECK-INST: stshh strm
// CHECK-ENCODING: encoding: [0x3f,0x96,0x01,0xd5]

prfm ir, [x0]
// CHECK-INST: prfm ir, [x0]
// CHECK-ENCODING: [0x18,0x00,0x80,0xf9]

prfm ir, [x2, #800]
// CHECK-INST: prfm ir, [x2, #800]
// CHECK-ENCODING: [0x58,0x90,0x81,0xf9]
