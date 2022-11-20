// RUN: rm -f %t
// RUN: llvm-mc -as-secure-log-file %t -triple x86_64-apple-darwin %s
// RUN: llvm-mc -as-secure-log-file %t -triple x86_64-apple-darwin %s
// RUN: FileCheck --input-file=%t %s
.secure_log_unique "foobar"

// CHECK: "foobar"
// CHECK-NEXT: "foobar"

