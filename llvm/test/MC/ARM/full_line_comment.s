// RUN: llvm-mc -triple arm-apple-darwin10 %s | FileCheck %s
# this is a full line comment starting at column 1
 # this starting at column 2

        .data
// CHECK: .long 0
.long 0
# .long 1 this line is commented out
