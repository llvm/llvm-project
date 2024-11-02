; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -force-vector-interleave=1 -S 2>&1 | FileCheck %s
; RUN: opt < %s -loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -force-vector-interleave=1 \
; RUN:   -mcpu=neoverse-n2 -S 2>&1 | FileCheck %s --check-prefix=CHECK-CPU-NEOVERSE-N2

target triple="aarch64-unknown-linux-gnu"

; CHECK: Found an estimated cost of 8 for VF vscale x 2 For instruction:   %add = fadd float %0, %sum.07
; CHECK: Found an estimated cost of 16 for VF vscale x 4 For instruction:   %add = fadd float %0, %sum.07
; CHECK-CPU-NEOVERSE-N2: Found an estimated cost of 4 for VF vscale x 2 For instruction:   %add = fadd float %0, %sum.07
; CHECK-CPU-NEOVERSE-N2: Found an estimated cost of 8 for VF vscale x 4 For instruction:   %add = fadd float %0, %sum.07

define float @fadd_strict32(float* noalias nocapture readonly %a, i64 %n) #0 {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %add
}


; CHECK: Found an estimated cost of 8 for VF vscale x 2 For instruction:   %add = fadd double %0, %sum.07
; CHECK-CPU-NEOVERSE-N2: Found an estimated cost of 4 for VF vscale x 2 For instruction:   %add = fadd double %0, %sum.07

define double @fadd_strict64(double* noalias nocapture readonly %a, i64 %n) #0 {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %iv
  %0 = load double, double* %arrayidx, align 4
  %add = fadd double %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret double %add
}

attributes #0 = { "target-features"="+sve" vscale_range(1, 16) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
