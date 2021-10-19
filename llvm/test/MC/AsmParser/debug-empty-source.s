// XFAIL: -aix
// UNSUPPORTED: -zos
// REQUIRES: default_triple
// RUN: llvm-mc %s -o -| FileCheck %s

.file 1 "dir1" "foo" source ""

# CHECK: .file {{.*}} source ""
