// RUN: llvm-mc -triple x86_64-apple-darwin10 %s | FileCheck %s

.long $1
// CHECK: .long ($1)
