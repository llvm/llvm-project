; RUN: opt < %s -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s

; CHECK: AccessFunction: ({0,+,%i2}<%outer.loop> + %unknown)
; CHECK-NEXT: failed to delinearize

; void foo(char A[], long i2, bool c) {
;   for (long i = 0; ; ++i) {
;     char *tmp = &A[i * i2];
;     if (c)
;       while (1)
;          *((float*)&tmp[arg << arg]) = 0;
;   }
; }

define void @foo(ptr %A, i64 %i2, i64 %arg, i1 %c) {
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %i414 = mul nsw i64 %outer.iv, %i2
  %tmp = getelementptr inbounds i8, ptr %A, i64 %i414
  br i1 %c, label %inner.preheader, label %outer.latch

inner.preheader:
  %unknown = shl i64 %arg, %arg
  %arrayidx = getelementptr inbounds i8, ptr %tmp, i64 %unknown
  br label %inner.loop

inner.loop:
  store float 0.000000e+00, ptr %arrayidx, align 4
  br label %inner.loop

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  br label %outer.loop
}
