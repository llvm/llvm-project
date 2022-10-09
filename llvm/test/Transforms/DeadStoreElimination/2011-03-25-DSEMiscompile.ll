; RUN: opt < %s -dse -S | FileCheck %s
; PR9561
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.8"

@A = external global [0 x i32]

declare ghccc void @Func2(ptr, ptr, ptr, i32)

define ghccc void @Func1(ptr noalias %Arg1, ptr noalias %Arg2, ptr %Arg3, i32 %Arg4) {
entry:
  store i32 add (i32 ptrtoint (ptr @A to i32), i32 1), ptr %Arg2
; CHECK: store i32 add (i32 ptrtoint (ptr @A to i32), i32 1), ptr %Arg2
  %ln2gz = getelementptr i32, ptr %Arg1, i32 14
  %ln2gB = load double, ptr %ln2gz
  %ln2gD = getelementptr i32, ptr %Arg2, i32 -3
  store double %ln2gB, ptr %ln2gD
; CHECK: store double %ln2gB, ptr %ln2gD
  tail call ghccc void @Func2(ptr %Arg1, ptr %Arg2, ptr %Arg3, i32 %Arg4) nounwind
  ret void
}
