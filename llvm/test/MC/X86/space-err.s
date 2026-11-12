// RUN: not llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux 2>&1 | FileCheck %s

// CHECK: space-err.s:[[@LINE+1]]:8: error: invalid number of bytes
.space -4
