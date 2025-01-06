; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -force-vector-interleave=1 -S 2>&1 | FileCheck %s --check-prefix=CHECK-VSCALE1
; RUN: opt < %s -passes=loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -force-vector-interleave=1 \
; RUN:   -mcpu=neoverse-v1 -S 2>&1 | FileCheck %s --check-prefix=CHECK-VSCALE2
; RUN: opt < %s -passes=loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -force-vector-interleave=1 \
; RUN:   -mcpu=neoverse-n2 -S 2>&1 | FileCheck %s --check-prefix=CHECK-VSCALE1

target triple="aarch64-unknown-linux-gnu"

; CHECK-VSCALE2-LABEL: LV: Checking a loop in 'fadd_strict32'
; CHECK-VSCALE2: Cost of 4 for VF vscale x 2:
; CHECK-VSCALE2:  in-loop reduction   %add = fadd float %0, %sum.07
; CHECK-VSCALE2: Cost of 8 for VF vscale x 4:
; CHECK-VSCALE2:  in-loop reduction   %add = fadd float %0, %sum.07
; CHECK-VSCALE1-LABEL: LV: Checking a loop in 'fadd_strict32'
; CHECK-VSCALE1: Cost of 2 for VF vscale x 2:
; CHECK-VSCALE1:  in-loop reduction   %add = fadd float %0, %sum.07
; CHECK-VSCALE1: Cost of 4 for VF vscale x 4:
; CHECK-VSCALE1:  in-loop reduction   %add = fadd float %0, %sum.07

define float @fadd_strict32(ptr noalias nocapture readonly %a, i64 %n) #0 {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %add
}


; CHECK-VSCALE2-LABEL: LV: Checking a loop in 'fadd_strict64'
; CHECK-VSCALE2: Cost of 4 for VF vscale x 2:
; CHECK-VSCALE2:  in-loop reduction   %add = fadd double %0, %sum.07
; CHECK-VSCALE1-LABEL: LV: Checking a loop in 'fadd_strict64'
; CHECK-VSCALE1: Cost of 2 for VF vscale x 2:
; CHECK-VSCALE1:  in-loop reduction   %add = fadd double %0, %sum.07

define double @fadd_strict64(ptr noalias nocapture readonly %a, i64 %n) #0 {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %iv
  %0 = load double, ptr %arrayidx, align 4
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
