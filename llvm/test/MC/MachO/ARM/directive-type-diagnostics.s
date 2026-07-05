// RUN: not llvm-mc -triple arm-apple -filetype asm -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple armeb-apple -filetype asm -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple thumb-apple -filetype asm -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple thumbeb-apple -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

        .type symbol 32
// CHECK: error: expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '%<type>' or "<type>"
// CHECK: .type symbol 32
// CHECK:              ^

