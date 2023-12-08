// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme,+fullfp16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=-neon < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme,+fullfp16 < %s \
// RUN:        | llvm-objdump --mattr=+fullfp16 -d - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme,+fullfp16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme,+fullfp16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Scalar FP instructions

fmulx h0, h1, h2
// CHECK-INST: fmulx h0, h1, h2
// CHECK-ENCODING: [0x20,0x1c,0x42,0x5e]
// CHECK-ERROR: instruction requires: fullfp16

frecps h0, h1, h2
// CHECK-INST: frecps h0, h1, h2
// CHECK-ENCODING: [0x20,0x3c,0x42,0x5e]
// CHECK-ERROR: instruction requires: fullfp16

frsqrts h0, h1, h2
// CHECK-INST: frsqrts h0, h1, h2
// CHECK-ENCODING: [0x20,0x3c,0xc2,0x5e]
// CHECK-ERROR: instruction requires: fullfp16

frecpe h0, h1
// CHECK-INST: frecpe h0, h1
// CHECK-ENCODING: [0x20,0xd8,0xf9,0x5e]
// CHECK-ERROR: instruction requires: fullfp16

frecpx h0, h1
// CHECK-INST: frecpx h0, h1
// CHECK-ENCODING: [0x20,0xf8,0xf9,0x5e]
// CHECK-ERROR: instruction requires: fullfp16

frsqrte h0, h1
// CHECK-INST: frsqrte h0, h1
// CHECK-ENCODING: [0x20,0xd8,0xf9,0x7e]
// CHECK-ERROR: instruction requires: fullfp16
