; RUN: llvm-dis < %s.bc | FileCheck %s

; Bitcode compatibility test for 'dynamic' parameter to llvm.objectsize.

define void @callit(i8* %ptr) {
; CHECK: %sz = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 true, i1 false, i1 true, i1 false)
  %sz = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 false, i1 true)
  ret void
}

; CHECK: declare i64 @llvm.objectsize.i64.p0(ptr, i1 immarg, i1 immarg, i1 immarg, i1)
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1)
