; frem is fmod, exact: a loop reduces |a| by |b| 2^j (each step a correctly
; rounded quotient below 2^24, truncated, and the remainder a - q d or
; a - (q - 1) d, whichever is in [0, d)) while the quotient could reach 2^24,
; then a last step by |b|; the sign of a, a for an infinite b, NaN for an
; infinite or NaN a or a zero or NaN b. Half is computed in float.
; RUN: llc -mtriple=agx -mcpu=agx-g14g -stop-after=agx-codegenprepare %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g14g -mattr=+ieee-fp32 -stop-after=agx-codegenprepare %s -o - \
; RUN:   | FileCheck %s --check-prefix=IEEE

; CHECK-LABEL: @f(
; CHECK: %[[AX:.*]] = call float @llvm.fabs.f32(float %a)
; CHECK: %[[AY:.*]] = call float @llvm.fabs.f32(float %b)
; CHECK: %[[NAN:.*]] = or i1
; CHECK: %[[YINF:.*]] = fcmp oeq float %[[AY]], +inf
; CHECK: br label %frem.loop
; CHECK: frem.loop:
; CHECK: %[[A:.*]] = phi float [ %{{.*}}, %{{.*}} ], [ %{{.*}}, %frem.reduce ]
; CHECK: br i1 %{{.*}}, label %frem.reduce, label %frem.done
; CHECK: frem.reduce:
; CHECK: fdiv float %[[A]],
; CHECK: call float @llvm.trunc.f32
; CHECK: br label %frem.loop
; CHECK: frem.done:
; CHECK: %[[Q:.*]] = call float @llvm.trunc.f32(float %{{.*}})
; CHECK: %[[QM1:.*]] = fsub float %[[Q]], 1.000000e+00
; CHECK: %[[LOWNEG:.*]] = fneg float %[[QM1]]
; CHECK: %[[LOW:.*]] = call float @llvm.fma.f32(float %[[LOWNEG]], float %[[AY]], float %[[A]])
; CHECK: %[[ISLOW:.*]] = fcmp olt float %[[LOW]], %[[AY]]
; CHECK: %[[QNEG:.*]] = fneg float %[[Q]]
; CHECK: %[[HIGH:.*]] = call float @llvm.fma.f32(float %[[QNEG]], float %[[AY]], float %[[A]])
; CHECK: %[[R:.*]] = select i1 %[[ISLOW]], float %[[LOW]], float %[[HIGH]]
; CHECK: %[[S:.*]] = call float @llvm.copysign.f32(float %[[R]], float %a)
; CHECK: %[[T:.*]] = select i1 %[[YINF]], float %a, float %[[S]]
; CHECK: %r = select i1 %[[NAN]], float +qnan, float %[[T]]
; CHECK-NOT: frem
define agx_kernel void @f(ptr addrspace(1) %p, float %a, float %b) {
  %r = frem float %a, %b
  store float %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @h(
; CHECK: fpext half %a to float
; CHECK: frem.loop
; CHECK: fptrunc float %{{.*}} to half
; CHECK-NOT: frem half
define agx_kernel void @h(ptr addrspace(1) %p, half %a, half %b) {
  %r = frem half %a, %b
  store half %r, ptr addrspace(1) %p
  ret void
}

; IEEE float denormals (ieee-fp32): a denormal b is normalized by 2^24 for
; the loop, with a last step by b itself.
; IEEE-LABEL: @f(
; IEEE: %[[AY:.*]] = call float @llvm.fabs.f32(float %b)
; IEEE: fmul float %[[AY]], f0x4B800000
; IEEE: frem.done:
; IEEE: fdiv float
; IEEE: fdiv float %{{.*}}, %[[AY]]
