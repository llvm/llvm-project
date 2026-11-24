; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

@a = global i8 42
@llvm.used = appending global [2 x ptr] [ptr @a, ptr null], section "llvm.metadata"
@llvm.compiler.used = appending global [2 x ptr] [ptr @a, ptr null], section "llvm.metadata"

; CHECK: invalid llvm.used member
; CHECK-NEXT: ptr null

; CHECK: invalid llvm.compiler.used member
; CHECK-NEXT: ptr null
