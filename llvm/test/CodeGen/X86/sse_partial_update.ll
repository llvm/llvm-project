; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+sse2 -mcpu=nehalem | FileCheck %s

; rdar: 12558838
; PR14221
; There is a mismatch between the intrinsic and the actual instruction.
; The actual instruction has a partial update of dest, while the intrinsic
; passes through the upper FP values. Here, we make sure the source and
; destination of rsqrtss are the same.
define void @t1(<4 x float> %a) nounwind uwtable ssp {
entry:
; CHECK: t1:
; CHECK: rsqrtss %xmm0, %xmm0
  %0 = tail call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %a) nounwind
  %a.addr.0.extract = extractelement <4 x float> %0, i32 0
  %conv = fpext float %a.addr.0.extract to double
  %a.addr.4.extract = extractelement <4 x float> %0, i32 1
  %conv3 = fpext float %a.addr.4.extract to double
  tail call void @callee(double %conv, double %conv3) nounwind
  ret void
}
declare void @callee(double, double)
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

define void @t2(<4 x float> %a) nounwind uwtable ssp {
entry:
; CHECK: t2:
; CHECK: rcpss %xmm0, %xmm0
  %0 = tail call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %a) nounwind
  %a.addr.0.extract = extractelement <4 x float> %0, i32 0
  %conv = fpext float %a.addr.0.extract to double
  %a.addr.4.extract = extractelement <4 x float> %0, i32 1
  %conv3 = fpext float %a.addr.4.extract to double
  tail call void @callee(double %conv, double %conv3) nounwind
  ret void
}
declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone
