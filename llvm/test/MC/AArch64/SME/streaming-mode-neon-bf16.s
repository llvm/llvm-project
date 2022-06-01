// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=-neon,+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=-neon < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=-neon,+sme < %s \
// RUN:        | llvm-objdump --mattr=-neon,+sme -d - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=-neon,+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=-neon,+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfcvt h5, s3
// CHECK-INST: bfcvt h5, s3
// CHECK-ENCODING: [0x65,0x40,0x63,0x1e]
// CHECK-ERROR: instruction requires: bf16 neon or sme
