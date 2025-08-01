; Partial expansion cases (still VP with parameter expansions).
; RUN: opt -passes=pre-isel-intrinsic-lowering --expandvp-override-evl-transform=Legal --expandvp-override-mask-transform=Legal -S < %s | FileCheck %s --check-prefix=LEGAL_LEGAL
; RUN: opt -passes=pre-isel-intrinsic-lowering --expandvp-override-evl-transform=Discard --expandvp-override-mask-transform=Legal -S < %s | FileCheck %s --check-prefix=DISCARD_LEGAL
; RUN: opt -passes=pre-isel-intrinsic-lowering --expandvp-override-evl-transform=Convert --expandvp-override-mask-transform=Legal -S < %s | FileCheck %s --check-prefix=CONVERT_LEGAL
; Full expansion cases (all expanded to non-VP).
; RUN: opt -passes=pre-isel-intrinsic-lowering --expandvp-override-evl-transform=Discard --expandvp-override-mask-transform=Convert -S < %s | FileCheck %s --check-prefix=ALL-CONVERT
; RUN: opt -passes=pre-isel-intrinsic-lowering -S < %s | FileCheck %s --check-prefix=ALL-CONVERT
; RUN: opt -passes=pre-isel-intrinsic-lowering --expandvp-override-evl-transform=Legal --expandvp-override-mask-transform=Convert -S < %s | FileCheck %s --check-prefix=ALL-CONVERT
; RUN: opt -passes=pre-isel-intrinsic-lowering --expandvp-override-evl-transform=Convert --expandvp-override-mask-transform=Convert -S < %s | FileCheck %s --check-prefix=ALL-CONVERT


; Fixed-width vectors
; Integer arith
declare <8 x i32> @llvm.vp.add.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sub.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.urem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.smax.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.smin.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.umax.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.umin.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
; Bit arith
declare <8 x i32> @llvm.vp.and.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.xor.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.or.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.shl.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
; Reductions
declare i32 @llvm.vp.reduce.add.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.mul.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.and.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.or.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.xor.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.smin.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.smax.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.umin.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.umax.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare float @llvm.vp.reduce.fmin.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.fmax.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.fminimum.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.fmaximum.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.fadd.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.fmul.v4f32(float, <4 x float>, <4 x i1>, i32)
; Comparisons
declare <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32>, <8 x i32>, metadata, <8 x i1>, i32)
declare <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float>, <8 x float>, metadata, <8 x i1>, i32)

; Fixed vector test function.
define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
  %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r7 = call <8 x i32> @llvm.vp.smax.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r8 = call <8 x i32> @llvm.vp.smin.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r9 = call <8 x i32> @llvm.vp.umax.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rA = call <8 x i32> @llvm.vp.umin.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rB = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rC = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rD = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rE = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rF = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r10 = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  ret void
}

; Scalable-width vectors
; Integer arith
declare <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.srem.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.urem.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.smax.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.smin.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.umax.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.umin.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
; Bit arith
declare <vscale x 4 x i32> @llvm.vp.and.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.xor.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.or.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.ashr.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.lshr.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.shl.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)

; Scalable vector test function.
define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
  %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r1 = call <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r2 = call <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r3 = call <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r4 = call <vscale x 4 x i32> @llvm.vp.srem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r5 = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r6 = call <vscale x 4 x i32> @llvm.vp.urem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r7 = call <vscale x 4 x i32> @llvm.vp.smax.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r8 = call <vscale x 4 x i32> @llvm.vp.smin.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r9 = call <vscale x 4 x i32> @llvm.vp.umax.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rA = call <vscale x 4 x i32> @llvm.vp.umin.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rB = call <vscale x 4 x i32> @llvm.vp.and.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rC = call <vscale x 4 x i32> @llvm.vp.or.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rD = call <vscale x 4 x i32> @llvm.vp.xor.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rE = call <vscale x 4 x i32> @llvm.vp.ashr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %rF = call <vscale x 4 x i32> @llvm.vp.lshr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  %r10 = call <vscale x 4 x i32> @llvm.vp.shl.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
  ret void
}

; Fixed vector reduce test function.
define void @test_vp_reduce_int_v4(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n) {
  %r0 = call i32 @llvm.vp.reduce.add.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r1 = call i32 @llvm.vp.reduce.mul.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r2 = call i32 @llvm.vp.reduce.and.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r3 = call i32 @llvm.vp.reduce.or.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r4 = call i32 @llvm.vp.reduce.xor.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r5 = call i32 @llvm.vp.reduce.smin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r6 = call i32 @llvm.vp.reduce.smax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r7 = call i32 @llvm.vp.reduce.umin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  %r8 = call i32 @llvm.vp.reduce.umax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
  ret void
}

define void @test_vp_reduce_fp_v4(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n) {
  %r0 = call float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r1 = call nnan float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r2 = call nnan ninf float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r3 = call float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r4 = call nnan float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r5 = call nnan ninf float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r6 = call float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r7 = call nnan float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r8 = call nnan ninf float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r9 = call float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r10 = call nnan float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r11 = call nnan ninf float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r12 = call float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r13 = call reassoc float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r14 = call float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  %r15 = call reassoc float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
  ret void
}

define void @test_vp_cmp_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x float> %f0, <8 x float> %f1, <8 x i1> %m, i32 %n) {
  %r0 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"eq", <8 x i1> %m, i32 %n)
  %r1 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"slt", <8 x i1> %m, i32 %n)
  %r2 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"oeq", <8 x i1> %m, i32 %n)
  %r3 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"ult", <8 x i1> %m, i32 %n)
  ret void
}

; All VP intrinsics have to be lowered into non-VP ops
; Convert %evl into %mask for non-speculatable VP intrinsics and emit the
; instruction+select idiom with a non-VP SIMD instruction.
;
; ALL-CONVERT-NOT: {{call.* @llvm.vp.add}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.sub}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.mul}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.sdiv}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.srem}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.udiv}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.urem}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.and}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.or}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.xor}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.ashr}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.lshr}}
; ALL-CONVERT-NOT: {{call.* @llvm.vp.shl}}
;
; ALL-CONVERT: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; ALL-CONVERT-NEXT:  %{{.*}} = add <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.*}} = sub <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.*}} = mul <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  [[NINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i64 0
; ALL-CONVERT-NEXT:  [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[EVLM:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; ALL-CONVERT-NEXT:  [[NEWM:%.+]] = and <8 x i1> [[EVLM]], %m
; ALL-CONVERT-NEXT:  [[SELONE:%.+]] = select <8 x i1> [[NEWM]], <8 x i32> %i1, <8 x i32> splat (i32 1)
; ALL-CONVERT-NEXT:  %{{.+}} = sdiv <8 x i32> %i0, [[SELONE]]
; ALL-CONVERT-NOT:   %{{.+}} = srem <8 x i32> %i0, %i1
; ALL-CONVERT:       %{{.+}} = srem <8 x i32> %i0, %{{.+}}
; ALL-CONVERT-NOT:   %{{.+}} = udiv <8 x i32> %i0, %i1
; ALL-CONVERT:       %{{.+}} = udiv <8 x i32> %i0, %{{.+}}
; ALL-CONVERT-NOT:   %{{.+}} = urem <8 x i32> %i0, %i1
; ALL-CONVERT:       %{{.+}} = urem <8 x i32> %i0, %{{.+}}
; ALL-CONVERT-NEXT:  %{{.+}} = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %i0, <8 x i32> %i1)
; ALL-CONVERT-NEXT:  %{{.+}} = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %i0, <8 x i32> %i1)
; ALL-CONVERT-NEXT:  %{{.+}} = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %i0, <8 x i32> %i1)
; ALL-CONVERT-NEXT:  %{{.+}} = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %i0, <8 x i32> %i1)
; ALL-CONVERT-NEXT:  %{{.+}} = and <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = or <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = xor <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = ashr <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = lshr <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT:  %{{.+}} = shl <8 x i32> %i0, %i1
; ALL-CONVERT:       ret void


; Check that reductions use the correct neutral element for masked-off elements
; ALL-CONVERT: define void @test_vp_reduce_int_v4(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n) {
; ALL-CONVERT-NEXT:  [[NINS:%.+]] = insertelement <4 x i32> poison, i32 %n, i64 0
; ALL-CONVERT-NEXT:  [[NSPLAT:%.+]] = shufflevector <4 x i32> [[NINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[EVLM:%.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, [[NSPLAT]]
; ALL-CONVERT-NEXT:  [[NEWM:%.+]] = and <4 x i1> [[EVLM]], %m
; ALL-CONVERT-NEXT:  [[ADD:%.+]] = select <4 x i1> [[NEWM]], <4 x i32> %vi, <4 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[ADD]])
; ALL-CONVERT-NEXT:  %{{.+}} = add i32 [[RED]], %start
; ALL-CONVERT:       [[MUL:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> splat (i32 1)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[MUL]])
; ALL-CONVERT-NEXT:  %{{.+}} = mul i32 [[RED]], %start
; ALL-CONVERT:       [[AND:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> splat (i32 -1)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> [[AND]])
; ALL-CONVERT-NEXT:  %{{.+}} = and i32 [[RED]], %start
; ALL-CONVERT:       [[OR:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> [[OR]])
; ALL-CONVERT-NEXT:  %{{.+}} = or i32 [[RED]], %start
; ALL-CONVERT:       [[XOR:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> [[XOR]])
; ALL-CONVERT-NEXT:  %{{.+}} = xor i32 [[RED]], %start
; ALL-CONVERT:       [[SMIN:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> splat (i32 2147483647)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> [[SMIN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call i32 @llvm.smin.i32(i32 [[RED]], i32 %start)
; ALL-CONVERT:       [[SMAX:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> splat (i32 -2147483648)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> [[SMAX]])
; ALL-CONVERT-NEXT:  %{{.+}} = call i32 @llvm.smax.i32(i32 [[RED]], i32 %start)
; ALL-CONVERT:       [[UMIN:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> splat (i32 -1)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> [[UMIN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call i32 @llvm.umin.i32(i32 [[RED]], i32 %start)
; ALL-CONVERT:       [[UMAX:%.+]] = select <4 x i1> %{{.+}}, <4 x i32> %vi, <4 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> [[UMAX]])
; ALL-CONVERT-NEXT:  %{{.+}} = call i32 @llvm.umax.i32(i32 [[RED]], i32 %start)
; ALL-CONVERT-NEXT:  ret void

; Check that reductions use the correct neutral element for masked-off elements
; ALL-CONVERT: define void @test_vp_reduce_fp_v4(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n) {
; ALL-CONVERT-NEXT:  [[NINS:%.+]] = insertelement <4 x i32> poison, i32 %n, i64 0
; ALL-CONVERT-NEXT:  [[NSPLAT:%.+]] = shufflevector <4 x i32> [[NINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; ALL-CONVERT-NEXT:  [[EVLM:%.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, [[NSPLAT]]
; ALL-CONVERT-NEXT:  [[NEWM:%.+]] = and <4 x i1> [[EVLM]], %m
; ALL-CONVERT-NEXT:  [[FMIN:%.+]] = select <4 x i1> [[NEWM]], <4 x float> %vf, <4 x float> splat (float 0x7FF8000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[FMIN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call float @llvm.minnum.f32(float [[RED]], float %f)
; ALL-CONVERT:       [[FMIN_NNAN:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0x7FF0000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[FMIN_NNAN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan float @llvm.minnum.f32(float [[RED]], float %f)
; ALL-CONVERT:       [[FMIN_NNAN_NINF:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0x47EFFFFFE0000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan ninf float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[FMIN_NNAN_NINF]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan ninf float @llvm.minnum.f32(float [[RED]], float %f)
; ALL-CONVERT:  [[FMAX:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0xFFF8000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[FMAX]])
; ALL-CONVERT-NEXT:  %{{.+}} = call float @llvm.maxnum.f32(float [[RED]], float %f)
; ALL-CONVERT:  [[FMAX_NNAN:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0xFFF0000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[FMAX_NNAN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan float @llvm.maxnum.f32(float [[RED]], float %f)
; ALL-CONVERT:  [[FMAX_NNAN_NINF:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0xC7EFFFFFE0000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan ninf float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[FMAX_NNAN_NINF]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan ninf float @llvm.maxnum.f32(float [[RED]], float %f)

; ALL-CONVERT:       [[FMINIMUM:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0x7FF0000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call float @llvm.vector.reduce.fminimum.v4f32(<4 x float> [[FMINIMUM]])
; ALL-CONVERT-NEXT:  %{{.+}} = call float @llvm.minimum.f32(float [[RED]], float %f)
; ALL-CONVERT:       [[FMINIMUM_NNAN:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0x7FF0000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan float @llvm.vector.reduce.fminimum.v4f32(<4 x float> [[FMINIMUM_NNAN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan float @llvm.minimum.f32(float [[RED]], float %f)
; ALL-CONVERT:       [[FMINIMUM_NNAN_NINF:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0x47EFFFFFE0000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan ninf float @llvm.vector.reduce.fminimum.v4f32(<4 x float> [[FMINIMUM_NNAN_NINF]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan ninf float @llvm.minimum.f32(float [[RED]], float %f)

; ALL-CONVERT:  [[FMAXIMUM:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0xFFF0000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call float @llvm.vector.reduce.fmaximum.v4f32(<4 x float> [[FMAXIMUM]])
; ALL-CONVERT-NEXT:  %{{.+}} = call float @llvm.maximum.f32(float [[RED]], float %f)
; ALL-CONVERT:  [[FMAXIMUM_NNAN:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0xFFF0000000000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan float @llvm.vector.reduce.fmaximum.v4f32(<4 x float> [[FMAXIMUM_NNAN]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan float @llvm.maximum.f32(float [[RED]], float %f)
; ALL-CONVERT:  [[FMAXIMUM_NNAN_NINF:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 0xC7EFFFFFE0000000)
; ALL-CONVERT-NEXT:  [[RED:%.+]] = call nnan ninf float @llvm.vector.reduce.fmaximum.v4f32(<4 x float> [[FMAXIMUM_NNAN_NINF]])
; ALL-CONVERT-NEXT:  %{{.+}} = call nnan ninf float @llvm.maximum.f32(float [[RED]], float %f)

; ALL-CONVERT:  [[FADD:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float -0.000000e+00)
; ALL-CONVERT-NEXT:  %{{.+}} = call float @llvm.vector.reduce.fadd.v4f32(float %f, <4 x float> [[FADD]])
; ALL-CONVERT:  [[FADD:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float -0.000000e+00)
; ALL-CONVERT-NEXT:  %{{.+}} = call reassoc float @llvm.vector.reduce.fadd.v4f32(float %f, <4 x float> [[FADD]])
; ALL-CONVERT:  [[FMUL:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 1.000000e+00)
; ALL-CONVERT-NEXT:  %{{.+}} = call float @llvm.vector.reduce.fmul.v4f32(float %f, <4 x float> [[FMUL]])
; ALL-CONVERT:  [[FMUL:%.+]] = select <4 x i1> %{{.+}}, <4 x float> %vf, <4 x float> splat (float 1.000000e+00)
; ALL-CONVERT-NEXT:  %{{.+}} = call reassoc float @llvm.vector.reduce.fmul.v4f32(float %f, <4 x float> [[FMUL]])
; ALL-CONVERT-NEXT:  ret void

; Check that comparisons use the correct condition codes
; ALL-CONVERT: define void @test_vp_cmp_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x float> %f0, <8 x float> %f1, <8 x i1> %m, i32 %n) {
; ALL-CONVERT-NEXT: %{{.+}} = icmp eq <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT: %{{.+}} = icmp slt <8 x i32> %i0, %i1
; ALL-CONVERT-NEXT: %{{.+}} = fcmp oeq <8 x float> %f0, %f1
; ALL-CONVERT-NEXT: %{{.+}} = fcmp ult <8 x float> %f0, %f1
; ALL-CONVERT-NEXT: ret void


; All legal - don't transform anything.

; LEGAL_LEGAL: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT:   %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r7 = call <8 x i32> @llvm.vp.smax.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r8 = call <8 x i32> @llvm.vp.smin.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r9 = call <8 x i32> @llvm.vp.umax.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rA = call <8 x i32> @llvm.vp.umin.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rB = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rC = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rD = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rE = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %rF = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   %r10 = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:   ret void

; LEGAL_LEGAL:define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT:  %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r1 = call <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r2 = call <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r3 = call <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r4 = call <vscale x 4 x i32> @llvm.vp.srem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r5 = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r6 = call <vscale x 4 x i32> @llvm.vp.urem.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r7 = call <vscale x 4 x i32> @llvm.vp.smax.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r8 = call <vscale x 4 x i32> @llvm.vp.smin.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r9 = call <vscale x 4 x i32> @llvm.vp.umax.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rA = call <vscale x 4 x i32> @llvm.vp.umin.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rB = call <vscale x 4 x i32> @llvm.vp.and.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rC = call <vscale x 4 x i32> @llvm.vp.or.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rD = call <vscale x 4 x i32> @llvm.vp.xor.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rE = call <vscale x 4 x i32> @llvm.vp.ashr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %rF = call <vscale x 4 x i32> @llvm.vp.lshr.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r10 = call <vscale x 4 x i32> @llvm.vp.shl.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  ret void

; LEGAL_LEGAL: define void @test_vp_reduce_int_v4(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT:  %r0 = call i32 @llvm.vp.reduce.add.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r1 = call i32 @llvm.vp.reduce.mul.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r2 = call i32 @llvm.vp.reduce.and.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r3 = call i32 @llvm.vp.reduce.or.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r4 = call i32 @llvm.vp.reduce.xor.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r5 = call i32 @llvm.vp.reduce.smin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r6 = call i32 @llvm.vp.reduce.smax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r7 = call i32 @llvm.vp.reduce.umin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  %r8 = call i32 @llvm.vp.reduce.umax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  ret void

; LEGAL_LEGAL: define void @test_vp_reduce_fp_v4(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT: %r0 = call float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r1 = call nnan float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r2 = call nnan ninf float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r3 = call float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r4 = call nnan float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r5 = call nnan ninf float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r6 = call float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r7 = call nnan float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r8 = call nnan ninf float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r9 = call float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r10 = call nnan float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r11 = call nnan ninf float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r12 = call float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r13 = call reassoc float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r14 = call float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r15 = call reassoc float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT:  ret void

; LEGAL_LEGAL: define void @test_vp_cmp_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x float> %f0, <8 x float> %f1, <8 x i1> %m, i32 %n) {
; LEGAL_LEGAL-NEXT: %r0 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"eq", <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r1 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"slt", <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r2 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"oeq", <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: %r3 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"ult", <8 x i1> %m, i32 %n)
; LEGAL_LEGAL-NEXT: ret void

; Drop %evl where possible else fold %evl into %mask (%evl Discard, %mask Legal)
;
; There is no caching yet in the ExpandVectorPredication pass and the %evl
; expansion code is emitted for every non-speculatable intrinsic again. Hence,
; only check that..
; (1) The %evl folding code and %mask are correct for the first
;     non-speculatable VP intrinsic.
; (2) All other non-speculatable VP intrinsics have a modified mask argument.
; (3) All speculatable VP intrinsics keep their %mask and %evl.
; (4) All VP intrinsics have an ineffective %evl parameter.

; DISCARD_LEGAL: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT:   %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   [[NSPLATINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i64 0
; DISCARD_LEGAL-NEXT:   [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NSPLATINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; DISCARD_LEGAL-NEXT:   [[EVLMASK:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; DISCARD_LEGAL-NEXT:   [[NEWMASK:%.+]] = and <8 x i1> [[EVLMASK]], %m
; DISCARD_LEGAL-NEXT:   %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> [[NEWMASK]], i32 8)
; DISCARD_LEGAL-NOT:    %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NOT:    %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NOT:    %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL:        %r7 = call <8 x i32> @llvm.vp.smax.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL:        %r8 = call <8 x i32> @llvm.vp.smin.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL:        %r9 = call <8 x i32> @llvm.vp.umax.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL:        %rA = call <8 x i32> @llvm.vp.umin.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rB = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rC = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rD = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rE = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %rF = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   %r10 = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT:   ret void

; TODO compute vscale only once and use caching.
; In the meantime, we only check for the correct vscale code for the first VP
; intrinsic and skip over it for all others.

; DISCARD_LEGAL: define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT: %vscale = call i32 @llvm.vscale.i32()
; DISCARD_LEGAL-NEXT: %scalable_size = mul nuw i32 %vscale, 4
; DISCARD_LEGAL-NEXT: %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %scalable_size)
; DISCARD_LEGAL:      %r1 = call <vscale x 4 x i32> @llvm.vp.sub.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %scalable_size{{.*}})
; DISCARD_LEGAL:      %r2 = call <vscale x 4 x i32> @llvm.vp.mul.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> %m, i32 %scalable_size{{.*}})
; DISCARD_LEGAL:      [[EVLM:%.+]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 %n)
; DISCARD_LEGAL:      [[NEWM:%.+]] = and <vscale x 4 x i1> [[EVLM]], %m
; DISCARD_LEGAL:      %r3 = call <vscale x 4 x i32> @llvm.vp.sdiv.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> [[NEWM]], i32 %scalable_size{{.*}})
; DISCARD_LEGAL-NOT:  %{{.+}} = call <vscale x 4 x i32> @llvm.vp.{{.*}}, i32 %n)
; DISCARD_LEGAL:      ret void

; DISCARD_LEGAL: define void @test_vp_reduce_int_v4(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT:  [[NSPLATINS:%.+]] = insertelement <4 x i32> poison, i32 %n, i64 0
; DISCARD_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <4 x i32> [[NSPLATINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; DISCARD_LEGAL-NEXT:  [[EVLMASK:%.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, [[NSPLAT]]
; DISCARD_LEGAL-NEXT:  [[NEWMASK:%.+]] = and <4 x i1> [[EVLMASK]], %m
; DISCARD_LEGAL-NEXT:  %r0 = call i32 @llvm.vp.reduce.add.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> [[NEWMASK]], i32 4)
; DISCARD_LEGAL-NOT:  %r1 = call i32 @llvm.vp.reduce.mul.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r2 = call i32 @llvm.vp.reduce.and.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r3 = call i32 @llvm.vp.reduce.or.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r4 = call i32 @llvm.vp.reduce.xor.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r5 = call i32 @llvm.vp.reduce.smin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r6 = call i32 @llvm.vp.reduce.smax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r7 = call i32 @llvm.vp.reduce.umin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r8 = call i32 @llvm.vp.reduce.umax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; DISCARD_LEGAL:      ret void

; DISCARD_LEGAL: define void @test_vp_reduce_fp_v4(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT:  [[NSPLATINS:%.+]] = insertelement <4 x i32> poison, i32 %n, i64 0
; DISCARD_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <4 x i32> [[NSPLATINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; DISCARD_LEGAL-NEXT:  [[EVLMASK:%.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, [[NSPLAT]]
; DISCARD_LEGAL-NEXT:  [[NEWMASK:%.+]] = and <4 x i1> [[EVLMASK]], %m
; DISCARD_LEGAL-NEXT:  %r0 = call float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> [[NEWMASK]], i32 4)
; DISCARD_LEGAL-NOT:  %r1 = call nnan float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r2 = call nnan ninf float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r3 = call float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r4 = call nnan float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r5 = call nnan ninf float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; DISCARD_LEGAL-NOT:  %r6 = call float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r7 = call nnan float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r8 = call nnan ninf float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r9 = call float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r10 = call nnan float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r11 = call nnan ninf float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r12 = call float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r13 = call reassoc float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r14 = call float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL-NOT:  %r15 = call reassoc float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n)
; DISCARD_LEGAL:      ret void

; DISCARD_LEGAL: define void @test_vp_cmp_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x float> %f0, <8 x float> %f1, <8 x i1> %m, i32 %n) {
; DISCARD_LEGAL-NEXT: %r0 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"eq", <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT: %r1 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"slt", <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT: %r2 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"oeq", <8 x i1> %m, i32 8)
; DISCARD_LEGAL-NEXT: %r3 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"ult", <8 x i1> %m, i32 8)

; Convert %evl into %mask everywhere (%evl Convert, %mask Legal)
;
; For the same reasons as in the (%evl Discard, %mask Legal) case only check that..
; (1) The %evl folding code and %mask are correct for the first VP intrinsic.
; (2) All other VP intrinsics have a modified mask argument.
; (3) All VP intrinsics have an ineffective %evl parameter.
;
; CONVERT_LEGAL: define void @test_vp_int_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x i32> %i2, <8 x i32> %f3, <8 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:  [[NINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i64 0
; CONVERT_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; CONVERT_LEGAL-NEXT:  [[EVLM:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; CONVERT_LEGAL-NEXT:  [[NEWM:%.+]] = and <8 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:  %{{.+}} = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> [[NEWM]], i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 8)
; CONVERT_LEGAL:       ret void

; Similar to %evl discard, %mask legal but make sure the first VP intrinsic has a legal expansion
; CONVERT_LEGAL: define void @test_vp_int_vscale(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i32> %i2, <vscale x 4 x i32> %f3, <vscale x 4 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:   [[EVLM:%.+]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i32(i32 0, i32 %n)
; CONVERT_LEGAL-NEXT:   [[NEWM:%.+]] = and <vscale x 4 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:   %vscale = call i32 @llvm.vscale.i32()
; CONVERT_LEGAL-NEXT:   %scalable_size = mul nuw i32 %vscale, 4
; CONVERT_LEGAL-NEXT:   %r0 = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %i0, <vscale x 4 x i32> %i1, <vscale x 4 x i1> [[NEWM]], i32 %scalable_size)
; CONVERT_LEGAL-NOT:    %{{.*}} = call <vscale x 4 x i32> @llvm.vp.{{.*}}, i32 %n)
; CONVERT_LEGAL:        ret void

; CONVERT_LEGAL: define void @test_vp_reduce_int_v4(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:  [[NINS:%.+]] = insertelement <4 x i32> poison, i32 %n, i64 0
; CONVERT_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <4 x i32> [[NINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CONVERT_LEGAL-NEXT:  [[EVLM:%.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, [[NSPLAT]]
; CONVERT_LEGAL-NEXT:  [[NEWM:%.+]] = and <4 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:  %{{.+}} = call i32 @llvm.vp.reduce.add.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> [[NEWM]], i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.mul.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.and.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.or.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.xor.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.smin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.smax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.umin.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call i32 @llvm.vp.reduce.umax.v4i32(i32 %start, <4 x i32> %vi, <4 x i1> %m, i32 4)
; CONVERT_LEGAL: ret void

; CONVERT_LEGAL: define void @test_vp_reduce_fp_v4(float %f, <4 x float> %vf, <4 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:  [[NINS:%.+]] = insertelement <4 x i32> poison, i32 %n, i64 0
; CONVERT_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <4 x i32> [[NINS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CONVERT_LEGAL-NEXT:  [[EVLM:%.+]] = icmp ult <4 x i32> <i32 0, i32 1, i32 2, i32 3>, [[NSPLAT]]
; CONVERT_LEGAL-NEXT:  [[NEWM:%.+]] = and <4 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:  %{{.+}} = call float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> [[NEWM]], i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan ninf float @llvm.vp.reduce.fmin.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan ninf float @llvm.vp.reduce.fmax.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> [[NEWM]], i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan ninf float @llvm.vp.reduce.fminimum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call nnan ninf float @llvm.vp.reduce.fmaximum.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call reassoc float @llvm.vp.reduce.fadd.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL-NOT:   %{{.+}} = call reassoc float @llvm.vp.reduce.fmul.v4f32(float %f, <4 x float> %vf, <4 x i1> %m, i32 4)
; CONVERT_LEGAL: ret void

; CONVERT_LEGAL: define void @test_vp_cmp_v8(<8 x i32> %i0, <8 x i32> %i1, <8 x float> %f0, <8 x float> %f1, <8 x i1> %m, i32 %n) {
; CONVERT_LEGAL-NEXT:  [[NINS:%.+]] = insertelement <8 x i32> poison, i32 %n, i64 0
; CONVERT_LEGAL-NEXT:  [[NSPLAT:%.+]] = shufflevector <8 x i32> [[NINS]], <8 x i32> poison, <8 x i32> zeroinitializer
; CONVERT_LEGAL-NEXT:  [[EVLM:%.+]] = icmp ult <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, [[NSPLAT]]
; CONVERT_LEGAL-NEXT:  [[NEWM:%.+]] = and <8 x i1> [[EVLM]], %m
; CONVERT_LEGAL-NEXT:  %{{.+}} = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"eq", <8 x i1> [[NEWM]], i32 8)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"eq", <8 x i1> %m, i32 %n)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"slt", <8 x i1> %m, i32 %n
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"oeq", <8 x i1> %m, i32 %n)
; CONVERT_LEGAL-NOT:   %{{.+}} = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"ult", <8 x i1> %m, i32 %n)
; CONVERT_LEGAL: ret void
