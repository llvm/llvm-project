; RUN: opt -passes=transform-warning -disable-output -pass-remarks-missed=transform-warning -pass-remarks-analysis=transform-warning < %s 2>&1 | FileCheck -allow-empty %s
;
; OpenMP SIMD loops are represented as annotated parallel loops. Keep
; transform-warning from emitting FailedRequestedVectorization for these loops.
;
; CHECK-NOT: FailedRequestedVectorization
; CHECK-NOT: loop not vectorized

define void @test(ptr %x, ptr %c) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %src.ptr = getelementptr inbounds double, ptr %c, i64 %i
  %v = load double, ptr %src.ptr, align 8, !llvm.access.group !1
  %dst.ptr = getelementptr inbounds double, ptr %x, i64 %i
  store double %v, ptr %dst.ptr, align 8, !llvm.access.group !1
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !2, !3}
!1 = distinct !{}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
!3 = !{!"llvm.loop.parallel_accesses", !1}
