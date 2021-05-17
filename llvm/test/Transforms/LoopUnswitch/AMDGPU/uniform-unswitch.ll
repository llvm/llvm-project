; RUN: opt -mtriple=amdgcn-- -O3 -S -enable-new-pm=1 %s | FileCheck %s
; XFAIL: *

; Check that loop unswitch happened and condition hoisted out of the loop.
; Condition is uniform so even targets with divergence should perform unswitching.

; This fails with the new pass manager:
; https://bugs.llvm.org/show_bug.cgi?id=48819
; The correct behaviour (allow uniform non-trivial branches to be
; unswitched on all targets) requires access to the function-level
; divergence analysis from a loop transform, which is currently not
; supported in the new pass manager.

; CHECK-LABEL: {{^}}define amdgpu_kernel void @uniform_unswitch
; CHECK: entry:
; CHECK-NEXT: [[LOOP_COND:%[a-z0-9]+]] = icmp
; CHECK-NEXT: [[IF_COND:%[a-z0-9]+]] = icmp eq i32 %x, 123456
; CHECK-NEXT: and i1 [[LOOP_COND]], [[IF_COND]]
; CHECK-NEXT: br i1

define amdgpu_kernel void @uniform_unswitch(i32 * nocapture %out, i32 %n, i32 %x) {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %cmp1 = icmp eq i32 %x, 123456
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.07 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32 * %out, i32 %i.07
  store i32 %i.07, i32 * %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
