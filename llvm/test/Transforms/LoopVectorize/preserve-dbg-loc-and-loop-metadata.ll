; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-width=4 -S | FileCheck %s -check-prefix DEBUGLOC
; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-width=4 -S --try-experimental-debuginfo-iterators | FileCheck %s -check-prefix DEBUGLOC
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; This test makes sure we don't duplicate the loop vectorizer's metadata
; while marking them as already vectorized (by setting width = 1), even
; at lower optimization levels, where no extra cleanup is done

; Check that the phi to resume the scalar part of the loop
; has Debug Location.
define void @_Z3fooPf(ptr %a) {
; DEBUGLOC-LABEL: define void @_Z3fooPf(
; DEBUGLOC: scalar.ph:
; DEBUGLOC-NEXT:    %bc.resume.val = phi {{.*}} !dbg ![[RESUMELOC:[0-9]+]]
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %p = load float, ptr %arrayidx, align 4
  %mul = fmul float %p, 2.000000e+00
  store float %mul, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

define void @widen_ptr_induction_dbg(ptr %start, ptr %end) {
; DEBUGLOC-LABEL: define void @widen_ptr_induction_dbg(
; DEBUGLOC: vector.body:
; DEBUGLOC-NEXT: = phi ptr {{.+}}, !dbg ![[PTRIVLOC:[0-9]+]]
; DEBUGLOC: = phi i64
;
; DEBUGLOC: loop:
; DEBUGLOC-NEXT: = phi ptr {{.+}}, !dbg ![[PTRIVLOC]]
;
entry:
  br label %loop

loop:
  %iv = phi ptr [ %start, %entry ], [ %iv.next, %loop ]
  %iv.next = getelementptr inbounds ptr, ptr %iv, i64 1
  store ptr %iv, ptr %iv, align 1
  %cmp.not = icmp eq ptr %iv.next, %end
  br i1 %cmp.not, label %exit, label %loop

exit:
  ret void
}


!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
; CHECK-NOT: !{metadata !"llvm.loop.vectorize.width", i32 4}
; CHECK: !{!"llvm.loop.isvectorized", i32 1}

; DEBUGLOC: ![[RESUMELOC]] = !DILocation(line: 2
; DEBUGLOC: ![[PTRIVLOC]] = !DILocation(line: 12
