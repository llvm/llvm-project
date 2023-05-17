; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=AST
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

; TODO: FIXME: Simplify the context.
; AST: if (n >= 1 && 0 == n <= -1)

; CHECK: entry:
; CHECK-NEXT: %0 = zext i32 %n to i64

; CHECK: polly.split_new_and_old:
; CHECK-NEXT:  %1 = sext i32 %n to i64
; CHECK-NEXT:  %2 = icmp sge i64 %1, 1
; CHECK-NEXT:  %3 = sext i32 %n to i64
; CHECK-NEXT:  %4 = icmp sle i64 %3, -1
; CHECK-NEXT:  %5 = sext i1 %4 to i64
; CHECK-NEXT:  %6 = icmp eq i64 0, %5
; CHECK-NEXT:  %7 = and i1 %2, %6
; CHECK-NEXT:  %polly.rtc.result = and i1 %7, true
; CHECK-NEXT:  br i1 %polly.rtc.result, label %polly.start, label %for.body4

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @init_array(i32 %n, ptr %data) {
entry:
  %0 = zext i32 %n to i64
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.body4 ], [ 0, %entry ]
  %.moved.to.for.body4 = mul i64 %0, %indvar1
  %1 = add i64 %.moved.to.for.body4, 0
  %arrayidx7 = getelementptr double, ptr %data, i64 %1
  store double undef, ptr %arrayidx7, align 8
  %indvar.next2 = add i64 %indvar1, 1
  br i1 false, label %for.body4, label %for.end10

for.end10:                                        ; preds = %for.body4
  ret void
}
