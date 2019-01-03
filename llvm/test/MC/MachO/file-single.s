// RUN: not llvm-mc -triple i386-apple-darwin9 %s -o /dev/null 2>&1 | FileCheck %s

// Previously this crashed MC.

// CHECK: error: target does not support '.file' without a number

        .file "dir/foo"
        nop
