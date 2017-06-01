; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-dis -o - %t | FileCheck %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; BCA-NOT: <GLOBALVAL_SUMMARY_BLOCK

; CHECK: @llvm.global_ctors = appending global
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @f }]

; CHECK: @g = internal global i8 42, !type !0
@g = internal global i8 42, !type !0

declare void @sink(i8*)

; CHECK: define internal void @f()
define internal void @f() {
  call void @sink(i8* @g)
  ret void
}

!0 = !{i32 0, !"typeid"}
