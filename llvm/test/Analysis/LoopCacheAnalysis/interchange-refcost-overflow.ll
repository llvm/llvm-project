; RUN: opt <  %s  -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck  %s

; For a loop with a very large iteration count, make sure the cost
; calculation does not overflow:
;
; void a() {
;   for (int c;; c += 2)
;     for (long d = 0; d < -3ULL; d += 2ULL)
;       for (long e = 0; e < -3ULL; e += 2ULL)
;         A[c][d][e] = 0;
; }

; CHECK: Loop 'outer.loop' has cost = 9223372036854775807
; CHECK: Loop 'middle.loop' has cost = 9223372036854775807
; CHECK: Loop 'inner.loop' has cost = 9223372036854775807

@A = local_unnamed_addr global [11 x [11 x [11 x i32]]] zeroinitializer, align 16

define void @foo() {
entry:
  br label %outer.loop

outer.loop:
  %indvars.iv = phi i64 [ %indvars.iv.next, %outer.loop.cleanup ], [ 0, %entry ]
  br label %middle.loop

outer.loop.cleanup:
  %indvars.iv.next = add i64 %indvars.iv, 2
  br label %outer.loop

middle.loop:
  %middle.iv = phi i64 [ %middle.iv.next, %middle.loop.cleanup ], [ 0, %outer.loop ]
  br label %inner.loop

middle.loop.cleanup:
  %middle.iv.next = add nuw i64 %middle.iv, 2
  %ec.middle = icmp ult i64 %middle.iv, -5
  br i1 %ec.middle, label %middle.loop, label %outer.loop.cleanup

inner.loop:
  %inner.iv = phi i64 [ 0, %middle.loop ], [ %add, %inner.loop ]
  %arrayidx3 = getelementptr inbounds [11 x [11 x [11 x i32]]], ptr @A, i64 0, i64 %indvars.iv, i64 %middle.iv, i64 %inner.iv
  store i32 0, ptr %arrayidx3, align 4
  %add = add nuw i64 %inner.iv, 2
  %cmp = icmp ult i64 %inner.iv, -5
  br i1 %cmp, label %inner.loop, label %middle.loop.cleanup
}
