; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@a = global i8 42
@llvm.used = appending global [2 x i8*] [i8* @a, i8* null], section "llvm.metadata"
@llvm.compiler.used = appending global [2 x i8*] [i8* @a, i8* null], section "llvm.metadata"

; CHECK: invalid llvm.used member
; CHECK-NEXT: i8* null

; CHECK: invalid llvm.compiler.used member
; CHECK-NEXT: i8* null
