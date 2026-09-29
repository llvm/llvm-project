// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o - 2>&1 | FileCheck %s

// CHECK: Undefined temporary
        .long .Lfoo
