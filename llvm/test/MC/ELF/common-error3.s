# RUN: not --crash llvm-mc -filetype=obj -triple x86_64-pc-linux %s 2>&1 | FileCheck %s

# CHECK: Symbol: C redeclared as different type
        .comm   C,4,4
        .comm   C,8,4
