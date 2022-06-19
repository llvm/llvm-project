; RUN: opt -S -passes=loop-vectorize < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @fn1
define void @fn1() {
entry-block:
  br label %middle

middle:
  %0 = phi {}* [ %3, %middle ], [ inttoptr (i64 0 to {}*), %entry-block ]
  %1 = bitcast {}* %0 to i8*
  %2 = getelementptr i8, i8* %1, i64 1
  %3 = bitcast i8* %2 to {}*
  %4 = icmp eq i8* %2, undef
  br i1 %4, label %exit, label %middle

; CHECK:      %[[phi:.*]] = phi {}* [ %3, %middle ], [ null, %entry-block ]
; CHECK-NEXT: %[[bc1:.*]] = bitcast {}* %[[phi]] to i8*
; CHECK-NEXT: %[[gep:.*]] = getelementptr i8, i8* %[[bc1]], i64 1
; CHECK-NEXT: %[[bc2:.*]] = bitcast i8* %[[gep]] to {}*
; CHECK-NEXT: %[[cmp:.*]] = icmp eq i8* %[[gep]], undef
; CHECK-NEXT: br i1 %[[cmp]],

exit:
  ret void
}
