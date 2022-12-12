// XFAIL: target={{.*}}-aix{{.*}}
// UNSUPPORTED: -zos
// REQUIRES: object-emission
// RUN: llvm-mc %s -o -| FileCheck %s

.file 1 "dir1" "foo" source ""

# CHECK: .file {{.*}} source ""
