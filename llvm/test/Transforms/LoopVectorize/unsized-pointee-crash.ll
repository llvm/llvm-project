; RUN: opt -S -passes=loop-vectorize < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @fn1
define void @fn1() {
entry:
  br label %for.body

for.body:
  %b.05 = phi i32 (...)* [ undef, %entry ], [ %1, %for.body ]
  %a.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = bitcast i32 (...)* %b.05 to i8*
  %add.ptr = getelementptr i8, i8* %0, i64 1
  %1 = bitcast i8* %add.ptr to i32 (...)*
; CHECK:      %[[cst:.*]] = bitcast i32 (...)* {{.*}} to i8*
; CHECK-NEXT: %[[gep:.*]] = getelementptr i8, i8* %[[cst]], i64 1
  %inc = add nsw i32 %a.04, 1
  %exitcond = icmp eq i32 %a.04, 63
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
