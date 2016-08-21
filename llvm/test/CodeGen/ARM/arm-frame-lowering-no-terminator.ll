; RUN: llc < %s
; Ensure that ARMFrameLowering can emit an epilogue when there's no terminator.
; This is the crasher from PR29072.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-linux-gnueabihf"

%t1 = type { [4 x float] }
%t3 = type { i32 (...)** }
%t2 = type { %t3, i8, %t1, %t1, float }

define internal void @foo(%t2* nocapture %this, %t1* nocapture readonly %triangle, i32 %partId, i32 %triangleIndex) {
entry:
  br i1 undef, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  %arrayidx.i = getelementptr inbounds %t1, %t1* %triangle, i32 0, i32 0, i32 0
  %0 = load float, float* %arrayidx.i, align 4
  %arrayidx5.i = getelementptr inbounds %t1, %t1* %triangle, i32 0, i32 0, i32 1
  %1 = load float, float* %arrayidx5.i, align 4
  %2 = load float, float* null, align 4
  %arrayidx11.i = getelementptr inbounds %t1, %t1* %triangle, i32 0, i32 0, i32 2
  %3 = load float, float* %arrayidx11.i, align 4
  %arrayidx13.i = getelementptr inbounds %t2, %t2* %this, i32 0, i32 2, i32 0, i32 2
  %4 = load float, float* %arrayidx13.i, align 4
  %arrayidx.i129 = getelementptr inbounds %t1, %t1* %triangle, i32 1, i32 0, i32 0
  %5 = load float, float* %arrayidx.i129, align 4
  %sub.i131 = fsub float %5, 0.000000e+00
  %arrayidx5.i132 = getelementptr inbounds %t1, %t1* %triangle, i32 1, i32 0, i32 1
  %6 = load float, float* %arrayidx5.i132, align 4
  %sub8.i134 = fsub float %6, %2
  %arrayidx11.i135 = getelementptr inbounds %t1, %t1* %triangle, i32 1, i32 0, i32 2
  %7 = load float, float* %arrayidx11.i135, align 4
  %sub14.i137 = fsub float %7, %4
  %arrayidx.i149 = getelementptr inbounds %t1, %t1* %triangle, i32 2, i32 0, i32 0
  %8 = load float, float* %arrayidx.i149, align 4
  %sub.i151 = fsub float %8, 0.000000e+00
  %arrayidx5.i152 = getelementptr inbounds %t1, %t1* %triangle, i32 2, i32 0, i32 1
  %9 = load float, float* %arrayidx5.i152, align 4
  %sub8.i154 = fsub float %9, %2
  %10 = load float, float* undef, align 4
  %sub14.i157 = fsub float %10, %4
  %mul.i = fmul float %sub8.i134, %sub14.i157
  %mul10.i = fmul float %sub14.i137, %sub8.i154
  %sub.i146 = fsub float %mul.i, %mul10.i
  %mul11.i = fmul float undef, %sub.i146
  %mul18.i = fmul float %sub14.i137, %sub.i151
  %mul23.i = fmul float %sub.i131, %sub14.i157
  %sub24.i = fsub float %mul18.i, %mul23.i
  %mul25.i = fmul float undef, %sub24.i
  %add.i148 = fadd float %mul11.i, %mul25.i
  %add40.i = fadd float undef, %add.i148
  %call.i = tail call float @fabsf(float %add40.i)
  %mul = fmul float %call.i, 2.500000e-01
  %add.i118 = fadd float %0, %5
  %add8.i121 = fadd float %1, %6
  %add14.i124 = fadd float %3, %7
  %add.i105 = fadd float %add.i118, %8
  %add8.i108 = fadd float %add8.i121, %9
  %add14.i111 = fadd float %add14.i124, %10
  %add.i93 = fadd float 0.000000e+00, %add.i105
  %add8.i96 = fadd float %2, %add8.i108
  %add14.i = fadd float %4, %add14.i111
  %mul.i.i = fmul float %add.i93, %mul
  %mul4.i.i = fmul float %add8.i96, %mul
  %mul8.i.i = fmul float %mul, %add14.i
  %arrayidx3.i = getelementptr inbounds %t2, %t2* %this, i32 0, i32 3, i32 0, i32 0
  %add.i = fadd float undef, %mul.i.i
  store float %add.i, float* %arrayidx3.i, align 4
  %arrayidx7.i86 = getelementptr inbounds %t2, %t2* %this, i32 0, i32 3, i32 0, i32 1
  %add8.i = fadd float %mul4.i.i, undef
  store float %add8.i, float* %arrayidx7.i86, align 4
  %arrayidx12.i = getelementptr inbounds %t2, %t2* %this, i32 0, i32 3, i32 0, i32 2
  %add13.i = fadd float %mul8.i.i, undef
  store float %add13.i, float* %arrayidx12.i, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %entry
  ret void
}

declare float @fabsf(float) readnone
