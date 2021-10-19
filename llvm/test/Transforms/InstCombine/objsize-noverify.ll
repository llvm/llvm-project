; Test objectsize bounds checking that won't verify until after -instcombine.
; RUN: opt < %s -disable-verify -instcombine -S | opt -S | FileCheck %s
; We need target data to get the sizes of the arrays and structures.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1) nounwind readonly

; CHECK-LABEL: @PR13390(
define i32 @PR13390(i1 %bool, i8* %a) {
entry:
  %cond = or i1 %bool, true
  br i1 %cond, label %return, label %xpto

xpto:
  %select = select i1 %bool, i8* %select, i8* %a
  %select2 = select i1 %bool, i8* %a, i8* %select2
  %0 = tail call i32 @llvm.objectsize.i32.p0i8(i8* %select, i1 true)
  %1 = tail call i32 @llvm.objectsize.i32.p0i8(i8* %select2, i1 true)
  %2 = add i32 %0, %1
; CHECK: ret i32 undef
  ret i32 %2

return:
  ret i32 42
}

define i32 @PR13390_logical(i1 %bool, i8* %a) {
entry:
  %cond = select i1 %bool, i1 true, i1 true
  br i1 %cond, label %return, label %xpto

xpto:
  %select = select i1 %bool, i8* %select, i8* %a
  %select2 = select i1 %bool, i8* %a, i8* %select2
  %0 = tail call i32 @llvm.objectsize.i32.p0i8(i8* %select, i1 true)
  %1 = tail call i32 @llvm.objectsize.i32.p0i8(i8* %select2, i1 true)
  %2 = add i32 %0, %1
; CHECK: ret i32 undef
  ret i32 %2

return:
  ret i32 42
}

; CHECK-LABEL: @PR13621(
define i32 @PR13621(i1 %bool) nounwind {
entry:
  %cond = or i1 %bool, true
  br i1 %cond, label %return, label %xpto

; technically reachable, but this malformed IR may appear as a result of constant propagation
xpto:
  %gep2 = getelementptr i8, i8* %gep, i32 1
  %gep = getelementptr i8, i8* %gep2, i32 1
  %o = call i32 @llvm.objectsize.i32.p0i8(i8* %gep, i1 true)
; CHECK: ret i32 undef
  ret i32 %o

return:
  ret i32 7
}

define i32 @PR13621_logical(i1 %bool) nounwind {
entry:
  %cond = select i1 %bool, i1 true, i1 true
  br i1 %cond, label %return, label %xpto

; technically reachable, but this malformed IR may appear as a result of constant propagation
xpto:
  %gep2 = getelementptr i8, i8* %gep, i32 1
  %gep = getelementptr i8, i8* %gep2, i32 1
  %o = call i32 @llvm.objectsize.i32.p0i8(i8* %gep, i1 true)
; CHECK: ret i32 undef
  ret i32 %o

return:
  ret i32 7
}
