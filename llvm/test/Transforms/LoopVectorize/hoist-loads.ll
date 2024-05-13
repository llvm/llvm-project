; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x float] zeroinitializer, align 16

; Make sure we can vectorize in the presence of hoistable conditional loads.
; CHECK-LABEL: @hoist_cond_load(
; CHECK: load <2 x float>

define void @hoist_cond_load() {
entry:
  br label %for.body
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end9 ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @A, i64 0, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds [1024 x float], ptr @B, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx2, align 4
  %cmp3 = fcmp oeq float %0, 0.000000e+00
  br i1 %cmp3, label %if.end9, label %if.else

if.else:
  %1 = load float, ptr %arrayidx, align 4
  br label %if.end9

if.end9:
  %tmp.0 = phi float [ %1, %if.else ], [ 0.000000e+00, %for.body ]
  store float %tmp.0, ptr %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; However, we can't hoist loads whose address we have not seen unconditionally
; accessed. One wide load is fine, but not the second.
; CHECK-LABEL: @dont_hoist_cond_load(
; CHECK: load <2 x float>
; CHECK-NOT: load <2 x float>

define void @dont_hoist_cond_load(ptr %a) {
entry:
  br label %for.body
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end9 ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr %a, i64 0, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds [1024 x float], ptr @B, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx2, align 4
  %cmp3 = fcmp oeq float %0, 0.000000e+00
  br i1 %cmp3, label %if.end9, label %if.else

if.else:
  %1 = load float, ptr %arrayidx, align 4
  br label %if.end9

if.end9:
  %tmp.0 = phi float [ %1, %if.else ], [ 0.000000e+00, %for.body ]
  store float %tmp.0, ptr %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}
