; REQUIRES: x86-registered-target

; The declaration is named \01___foo and the definition __foo, both the symbol
; ___foo. The definition must be kept when the declaration is read first.

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/global-prefix-escaped-name.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.out -r=%t1.bc,_use,px -r=%t1.bc,___foo, -r=%t2.bc,___foo,p
; RUN: llvm-nm %t.out.1 %t.out.2 | FileCheck %s

; CHECK-DAG: T ___foo
; CHECK-DAG: T _use

target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

declare i32 @"\01___foo"(i32)

define dso_local i32 @use(i32 %x) {
  %r = call i32 @"\01___foo"(i32 %x)
  ret i32 %r
}
