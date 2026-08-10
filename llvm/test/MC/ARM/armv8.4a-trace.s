// RUN: llvm-mc -triple arm -mattr=+v8.4a -show-encoding < %s | FileCheck %s  --check-prefix=CHECK-A32
// RUN: llvm-mc -triple thumb -mattr=+v8.4a -show-encoding < %s | FileCheck %s  --check-prefix=CHECK-T32
// RUN: not llvm-mc -triple arm -mattr=-v8.4a -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-V84

// RUN: llvm-mc -filetype=obj -triple arm -mattr=+v8.4a < %s \
// RUN: | llvm-objdump --triple=arm --mattr=+v8.4a -r -d - \
// RUN: | FileCheck -check-prefixes=CHECK-OBJDUMP %s
// RUN: llvm-mc -filetype=obj -triple thumb -mattr=+v8.4a < %s \
// RUN: | llvm-objdump --triple=thumb --mattr=+v8.4a -r -d - \
// RUN: | FileCheck -check-prefixes=CHECK-OBJDUMP %s

tsb csync

//CHECK-A32: tsb csync                   @ encoding: [0x12,0xf0,0x20,0xe3]
//CHECK-T32: tsb csync                   @ encoding: [0xaf,0xf3,0x12,0x80]
//CHECK-OBJDUMP: tsb csync

//CHECK-NO-V84: error: invalid instruction
//CHECK-NO-V84: tsb csync
//CHECK-NO-V84: ^
