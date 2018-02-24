// RUN: llvm-mc %s -o -| FileCheck %s

.file 1 "dir1" "foo" source ""
.loc 1 1 0
nop

# CHECK: .file {{.*}} source ""
