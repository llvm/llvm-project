; RUN: llc -mtriple=aarch64 -stop-after=loop-reduce < %s | FileCheck %s

declare void @foo(i64)

; Verify that redundant adds or geps aren't inserted by LSR.
; CHECK-LABEL: @bar(
define void @bar(ptr %A) {
entry:
  br label %while.cond

while.cond:
; CHECK-LABEL: while.cond:
; CHECK-NOT: add i64 %lsr.iv, 1
; CHECK-LABEL: land.rhs:
; CHECK: getelementptr i8, ptr %lsr.iv, i64 -8
; CHECK-NOT: getelementptr i8, ptr %lsr.iv, i64 -8
; CHECK-NOT: add i64, %lsr.iv, 1
  %indvars.iv28 = phi i64 [ %indvars.iv.next29, %land.rhs ], [ 50, %entry ]
  %cmp = icmp sgt i64 %indvars.iv28, 0
  br i1 %cmp, label %land.rhs, label %while.end

land.rhs:
  %indvars.iv.next29 = add nsw i64 %indvars.iv28, -1
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %indvars.iv.next29
  %Aload = load double, ptr %arrayidx, align 8
  %cmp1 = fcmp oeq double %Aload, 0.000000e+00
  br i1 %cmp1, label %while.cond, label %if.end

while.end:
  %indvars.iv28.lcssa = phi i64 [ %indvars.iv28, %while.cond ]
  tail call void @foo(i64 %indvars.iv28.lcssa)
  br label %if.end

if.end:
  ret void
}
