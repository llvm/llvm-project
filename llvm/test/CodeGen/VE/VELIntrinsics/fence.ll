; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: nounwind mustprogress
define void @_Z6fenceiv() {
; CHECK: fencei
  tail call void @llvm.ve.vl.fencei()
  ret void
}

; Function Attrs: nounwind
declare void @llvm.ve.vl.fencei()

; Function Attrs: nounwind mustprogress
define void @_Z7fencem3v() {
; CHECK: fencem 3
  tail call void @llvm.ve.vl.fencem.s(i32 3)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.ve.vl.fencem.s(i32)

; Function Attrs: nounwind mustprogress
define void @_Z7fencec7v() {
; CHECK: fencec 7
  tail call void @llvm.ve.vl.fencec.s(i32 7)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.ve.vl.fencec.s(i32)
