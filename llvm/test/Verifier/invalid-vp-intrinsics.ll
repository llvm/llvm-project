; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare <4 x i32> @llvm.vp.fptosi.v4i32.v8f32(<8 x float>, <4 x i1>, i32)
declare <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float>, <4 x float>, metadata, <4 x i1>, i32)
declare <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32>, <4 x i32>, metadata, <4 x i1>, i32)

; CHECK: VP cast intrinsic first argument and result vector lengths must be equal
; CHECK-NEXT: %r0 = call <4 x i32>

define void @test_vp_fptosi(<8 x float> %src, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i32> @llvm.vp.fptosi.v4i32.v8f32(<8 x float> %src, <4 x i1> %m, i32 %n)
  ret void
}

; CHECK: invalid predicate for VP FP comparison intrinsic
; CHECK-NEXT: %r0 = call <4 x i1> @llvm.vp.fcmp.v4f32
; CHECK: invalid predicate for VP FP comparison intrinsic
; CHECK-NEXT: %r1 = call <4 x i1> @llvm.vp.fcmp.v4f32

define void @test_vp_fcmp(<4 x float> %a, <4 x float> %b, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float> %a, <4 x float> %b, metadata !"bad", <4 x i1> %m, i32 %n)
  %r1 = call <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float> %a, <4 x float> %b, metadata !"eq", <4 x i1> %m, i32 %n)
  ret void
}

; CHECK: invalid predicate for VP integer comparison intrinsic
; CHECK-NEXT: %r0 = call <4 x i1> @llvm.vp.icmp.v4i32
; CHECK: invalid predicate for VP integer comparison intrinsic
; CHECK-NEXT: %r1 = call <4 x i1> @llvm.vp.icmp.v4i32

define void @test_vp_icmp(<4 x i32> %a, <4 x i32> %b, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> %a, <4 x i32> %b, metadata !"bad", <4 x i1> %m, i32 %n)
  %r1 = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> %a, <4 x i32> %b, metadata !"oeq", <4 x i1> %m, i32 %n)
  ret void
}
