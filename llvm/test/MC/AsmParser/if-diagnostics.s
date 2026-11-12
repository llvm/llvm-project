// RUN: not llvm-mc -triple i386 %s -o /dev/null 2>&1 | FileCheck %s

.if
.endif

// CHECK: error: unknown token in expression
// CHECK: .if
// CHECK:   ^

.ifeq 0, 3
.endif

// CHECK:error: expected newline
// CHECK: .ifeq 0, 3
// CHECK:        ^

.iflt "string1"
.endif

// CHECK: error: expected absolute expression
// CHECK: .iflt "string1"
// CHECK:       ^

.ifge test
.endif

// CHECK: error: expected absolute expression
// CHECK: .ifge test
// CHECK:       ^
