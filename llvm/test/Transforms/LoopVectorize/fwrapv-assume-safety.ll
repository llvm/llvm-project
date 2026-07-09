; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S < %s | FileCheck %s

; Verify that a loop with a non-unit stride (stride = 3) and no 'nsw' flag on the
; induction variable (e.g. from compilation with -fwrapv) can be vectorized
; by generating a runtime SCEV check to verify the lack of overflow/wrapping.

define void @fwrapv_stride3(ptr noalias %x, i32 %l, i32 %u) {
; CHECK-LABEL: @fwrapv_stride3(
; CHECK:       vector.scevcheck:
; CHECK:         %ident.check = icmp sgt i32 %u, 2147483645
; CHECK:         %mul = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 3, i32 {{.*}})
; CHECK:         %mul.result = extractvalue { i32, i1 } %mul, 0
; CHECK:         %mul.overflow = extractvalue { i32, i1 } %mul, 1
; CHECK:         [[IDENT:%.*]] = add i32 %l, %mul.result
; CHECK:         [[OVERFLOW:%.*]] = icmp slt i32 [[IDENT]], %l
; CHECK:         [[OVERFLOW2:%.*]] = or i1 [[OVERFLOW]], %mul.overflow
; CHECK:         [[CHECK:%.*]] = or i1 %ident.check, [[OVERFLOW2]]
; CHECK:         br i1 [[CHECK]], label %scalar.ph, label %vector.ph
; CHECK:       vector.body:
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
; CHECK:       scalar.ph:
; CHECK:         [[RESUME_PHI:%.*]] = phi i32
; CHECK:         br label %loop.body

entry:
  %cmp1 = icmp slt i32 %l, %u
  br i1 %cmp1, label %loop.body, label %exit

loop.body:
  %i = phi i32 [ %l, %entry ], [ %i.next, %loop.body ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, ptr %x, i64 %idxprom
  %val = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %val, 1
  store i32 %inc, ptr %arrayidx, align 4
  %i.next = add i32 %i, 3
  %cmp = icmp slt i32 %i.next, %u
  br i1 %cmp, label %loop.body, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
