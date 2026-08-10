; RUN: llvm-link %p/alignment.ll %p/Inputs/alignment.ll -S | FileCheck %s
; RUN: llvm-link %p/Inputs/alignment.ll %p/alignment.ll -S | FileCheck %s


@A = weak global i32 7, align 4
; CHECK-DAG: @A = global i32 7, align 8

@B = weak global i32 7, align 8
; CHECK-DAG: @B = global i32 7, align 4

define weak void @C() align 4 {
  ret void
}
; CHECK-DAG: define void @C() align 8 {

define weak void @D() align 8 {
  ret void
}
; CHECK-DAG: define void @D() align 4 {

@E = common global i32 0, align 4
; CHECK-DAG: @E = common global i32 0, align 8
