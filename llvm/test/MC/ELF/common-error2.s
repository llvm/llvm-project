// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux < %s 2>&1 | FileCheck %s

        .set    A,C
        .comm   C,4,4

// CHECK: Common symbol C cannot be used in assignment expr
