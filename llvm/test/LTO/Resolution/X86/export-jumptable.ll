; Test that we do not internalize functions that appear in the CFI jump table in
; the full LTO object file; any such functions will be referenced by the jump
; table.

; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-lto2 run -o %t2 -r %t,f1,p -r %t,f2,p -r %t,_start,px %t -save-temps
; RUN: llvm-dis %t2.1.2.internalize.bc -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define void @f1()
define void @f1() !type !0 {
  ret void
}

; CHECK: define internal void @f2()
define void @f2() !type !1 {
  ret void
}

define i1 @_start(i8* %p) {
  %1 = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  call void @f1()
  call void @f2()
  ret i1 %1
}

declare i1 @llvm.type.test(i8*, metadata)

!0 = !{i64 0, !"typeid1"}
!1 = !{i64 0, !"typeid2"}
