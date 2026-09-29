; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s


; CHECK: Just Ref: call void @ro() <-> call void @f0()

declare void @f0()
declare void @ro() readonly

define void @test0() {
  call void @f0()
  call void @ro()
  ret void
}

; CHECK: NoModRef:   call void @llvm.memset.p0.i64(ptr @A, i8 0, i64 1, i1 false) <->   call void @llvm.memset.p0.i64(ptr @B, i8 0, i64 1, i1 false)
; CHECK: NoModRef:   call void @llvm.memset.p0.i64(ptr @B, i8 0, i64 1, i1 false) <->   call void @llvm.memset.p0.i64(ptr @A, i8 0, i64 1, i1 false)

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind

@A = external global i8
@B = external global i8
define void @test1() {
  call void @llvm.memset.p0.i64(ptr @A, i8 0, i64 1, i1 false)
  call void @llvm.memset.p0.i64(ptr @B, i8 0, i64 1, i1 false)
  ret void
}
