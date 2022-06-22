; RUN: llc -mtriple=i686-- < %s
; RUN: llc -mtriple=x86_64-- < %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"

define float @MakeSphere(float %theta.079) nounwind {
entry:
  %add36 = fadd float %theta.079, undef
  %call = call float @cosf(float %theta.079) nounwind readnone
  %call45 = call float @sinf(float %theta.079) nounwind readnone
  %call37 = call float @sinf(float %add36) nounwind readnone
  store float %call, ptr undef, align 8
  store float %call37, ptr undef, align 8
  store float %call45, ptr undef, align 8
  ret float %add36
}

define hidden fastcc void @unroll_loop(i64 %storemerge32129) nounwind {
entry:
  call fastcc void @copy_rtx() nounwind
  call fastcc void @copy_rtx() nounwind
  %tmp225 = alloca i8, i64 %storemerge32129, align 8 ; [#uses=0 type=ptr]
  %cmp651201 = icmp slt i64 %storemerge32129, 0   ; [#uses=1 type=i1]
  br i1 %cmp651201, label %for.body653.lr.ph, label %if.end638.for.end659_crit_edge

for.body653.lr.ph:                                ; preds = %entry
  unreachable

if.end638.for.end659_crit_edge:                   ; preds = %entry
  unreachable
}

declare float @cosf(float) nounwind readnone
declare float @sinf(float) nounwind readnone
declare hidden fastcc void @copy_rtx() nounwind
