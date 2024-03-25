;; Tag this 'XFAIL' because we need a few more TTIs and ISels.
; XFAIL: *
; RUN: opt --mtriple=loongarch64 --passes=loop-data-prefetch -loongarch-enable-loop-data-prefetch -S < %s | FileCheck %s

define void @foo(ptr %a, ptr %b) {
entry:
  br label %for.body

; CHECK: for.body:
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %indvars.iv
; CHECK: call void @llvm.prefetch
  %0 = load double, ptr %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  store double %add, ptr %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

; CHECK: for.end:
for.end:                                          ; preds = %for.body
  ret void
}
