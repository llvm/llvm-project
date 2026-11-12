// RUN: llvm-mc -triple i686-pc-win32 %s | FileCheck %s

// check that we quote the output of .safeseh

.safeseh "\\foo"
// CHECK: .safeseh "\\foo"
