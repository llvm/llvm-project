; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-widen-divrem-via-safe-divisor=0 -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-width=4 -force-widen-divrem-via-safe-divisor=0 -S | FileCheck %s -check-prefix DEBUGLOC
; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-width=4 -force-widen-divrem-via-safe-divisor=0 -S --try-experimental-debuginfo-iterators | FileCheck %s -check-prefix DEBUGLOC
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

define void @predicated_phi_dbg(i64 %n, ptr %x) {
; DEBUGLOC-LABEL: define void @predicated_phi_dbg(
; DEBUGLOC: pred.udiv.continue{{.+}}:
; DEBUGLOC-NEXT:   = phi <4 x i64> {{.+}}, !dbg [[PREDPHILOC:![0-9]+]]
;
; DEBUGLOC: for.body:
; DEBUGLOC:   %tmp4 = udiv i64 %n, %i, !dbg [[PREDPHILOC]]
;
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %cmp = icmp ult i64 %i, 5
  br i1 %cmp, label %if.then, label %for.inc

if.then:
  %tmp4 = udiv i64 %n, %i
  br label %for.inc

for.inc:
  %d = phi i64 [ 0, %for.body ], [ %tmp4, %if.then ]
  %idx = getelementptr i64, ptr %x, i64 %i
  store i64 %d, ptr %idx
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

define void @scalar_cast_dbg(ptr nocapture %a, i32 %start, i64 %k) {
; DEBUGLOC-LABEL: define void @scalar_cast_dbg(
; DEBUGLOC:   = trunc i64 %index to i32, !dbg [[CASTLOC:![0-9]+]]
;
; DEBUGLOC: loop:
; DEBUGLOC:   %trunc.iv = trunc i64 %iv to i32, !dbg [[CASTLOC]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %trunc.iv = trunc i64 %iv to i32
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %trunc.iv
  store i32 %trunc.iv, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %k
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @widen_intrinsic_dbg(i64 %n, ptr %y, ptr %x) {
; DEBUGLOC-LABEL: define void @widen_intrinsic_dbg(
; DEBUGLOC: vector.body:
; DEBUGLOC:   = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}), !dbg ![[INTRINSIC_LOC:[0-9]+]]
; DEBUGLOC: loop:
; DEBUGLOC:   = call float @llvm.sqrt.f32(float %{{.+}}), !dbg ![[INTRINSIC_LOC]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.y = getelementptr inbounds float, ptr %y, i64 %iv
  %load = load float, ptr %gep.y, align 4
  %call = call float @llvm.sqrt.f32(float %load)
  %gep.x = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %gep.x, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
; CHECK-NOT: !{metadata !"llvm.loop.vectorize.width", i32 4}
; CHECK: !{!"llvm.loop.isvectorized", i32 1}

; DEBUGLOC: ![[RESUMELOC]] = !DILocation(line: 2
; DEBUGLOC: ![[PTRIVLOC]] = !DILocation(line: 12
; DEBUGLOC: ![[INTRINSIC_LOC]] = !DILocation(line: 44
