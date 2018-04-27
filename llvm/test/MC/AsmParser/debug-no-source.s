// RUN: llvm-mc %s | FileCheck %s
// REQUIRES: default_triple

.file 1 "dir1/foo"

# CHECK-NOT: .file {{.*}} source
