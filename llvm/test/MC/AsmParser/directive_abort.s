// RUN: not llvm-mc -filetype=obj -triple x86_64 %s 2>&1 -o /dev/null | FileCheck %s

.abort
// CHECK:      [[#@LINE-1]]:1: error: .abort detected. Assembly stopping
// CHECK-NEXT: abort

.abort "abort message"
// CHECK:      [[#@LINE-1]]:1: error: .abort '"abort message"' detected. Assembly stopping
// CHECK-NEXT: abort
