; RUN: opt -passes=transform-warning -disable-output -pass-remarks-missed=transform-warning -pass-remarks-analysis=transform-warning < %s 2>&1 | FileCheck %s
; RUN: opt -passes=transform-warning -disable-output -pass-remarks-output=%t.yaml < %s
; RUN: cat %t.yaml | FileCheck %s --check-prefix=YAML

; OpenMP SIMD loops are annotated parallel loops and may legally run without
; vectorization, so transform-warning must not report them. A forced loop that
; is not annotated parallel still gets the warning.

; CHECK-NOT: loop not vectorized
; CHECK: warning: {{.*}}loop not vectorized
; CHECK-NOT: warning:

; YAML-NOT: Function:        test_omp_simd
; YAML: Function:        test_forced

define void @test_omp_simd(ptr %x, ptr %c) {
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

define void @test_forced(ptr %x, ptr %c) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %src.ptr = getelementptr inbounds double, ptr %c, i64 %i
  %v = load double, ptr %src.ptr, align 8
  %dst.ptr = getelementptr inbounds double, ptr %x, i64 %i
  store double %v, ptr %dst.ptr, align 8
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop, !llvm.loop !2

exit:
  ret void
}

!0 = distinct !{!0, !3, !4}
!1 = distinct !{}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.vectorize.enable"}
!4 = !{!"llvm.loop.parallel_accesses", !1}
