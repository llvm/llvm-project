// RUN: llvm-mc %s -o -| FileCheck %s
// REQUIRES: default_triple

.file 1 "dir1" "foo" source ""

# CHECK: .file {{.*}} source ""
