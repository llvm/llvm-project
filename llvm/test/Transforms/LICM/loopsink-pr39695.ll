; RUN: opt -S -verify-memoryssa -loop-sink < %s | FileCheck %s
; RUN: opt -S -verify-memoryssa -aa-pipeline=basic-aa -passes=loop-sink < %s | FileCheck %s

; The load instruction should not be sunk into following loop.
; CHECK:      @foo
; CHECK-NEXT: entry
; CHECK-NEXT: %ptr = load ptr, ptr %pp, align 8
; CHECK-NEXT: store ptr null, ptr %pp, align 8

define i32 @foo(i32 %n, ptr %pp) !prof !0 {
entry:
  %ptr = load ptr, ptr %pp, align 8
  store ptr null, ptr %pp, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end, !prof !1

for.body:                                         ; preds = %for.cond
  %0 = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i8, ptr %ptr, i64 %0
  %1 = load i8, ptr %arrayidx, align 1
  %or19 = call i8 @llvm.bitreverse.i8(i8 %1)
  %v = sext i8 %or19 to i32
  %inc = add i32 %i.0, %v
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 %i.0
}

declare i8 @llvm.bitreverse.i8(i8) #0
attributes #0 = { nounwind readnone speculatable }

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
