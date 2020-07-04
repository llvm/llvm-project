; RUN: not opt -verify < %s 2>&1 | FileCheck %s

declare void @llvm.assume(i1)

define void @func(i32* %P, i32 %P1, i32* %P2, i32* %P3) {
; CHECK: tags must be valid attribute names
  call void @llvm.assume(i1 true) ["adazdazd"()]
; CHECK: the second argument should be a constant integral value
  call void @llvm.assume(i1 true) ["align"(i32* %P, i32 %P1)]
; CHECK: to many arguments
  call void @llvm.assume(i1 true) ["align"(i32* %P, i32 8, i32 8)]
; CHECK: this attribute should have 2 arguments
  call void @llvm.assume(i1 true) ["align"(i32* %P)]
; CHECK: this attribute has no argument
  call void @llvm.assume(i1 true) ["align"(i32* %P, i32 4), "cold"(i32* %P)]
; CHECK: this attribute should have one argument
  call void @llvm.assume(i1 true) ["noalias"()]
  ret void
}
