; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-unknown < %s  | \
; RUN:   FileCheck %s

@glob = local_unnamed_addr global <4 x float> zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @test(float %a, ptr nocapture readonly %b) {
; CHECK-LABEL: test
; CHECK: xscvdpspn [[REG:[0-9]+]], 1
; CHECK: xxspltw {{[0-9]+}}, [[REG]], 0
entry:
  %splat.splatinsert = insertelement <4 x float> undef, float %a, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  %0 = load <4 x float>, ptr %b, align 4
  %mul = fmul <4 x float> %splat.splat, %0
  store <4 x float> %mul, ptr @glob, align 4
  ret void
}
