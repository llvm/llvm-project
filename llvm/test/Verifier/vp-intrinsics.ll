; RUN: opt -passes=verify --disable-output %s

define void @test_vp_int(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n) {
  %r0 = call <8 x i32> @llvm.vp.add.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r1 = call <8 x i32> @llvm.vp.sub.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r2 = call <8 x i32> @llvm.vp.mul.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r3 = call <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r4 = call <8 x i32> @llvm.vp.srem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r5 = call <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r6 = call <8 x i32> @llvm.vp.urem.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r7 = call <8 x i32> @llvm.vp.and.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r8 = call <8 x i32> @llvm.vp.or.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %r9 = call <8 x i32> @llvm.vp.xor.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rA = call <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rB = call <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  %rC = call <8 x i32> @llvm.vp.shl.v8i32(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %n)
  ret void
}


define void @test_vp_fp(<8 x double> %f0, <8 x double> %f1, <8 x i1> %m, i32 %n) {
  %r0 = call <8 x double> @llvm.vp.fadd.v8f64(<8 x double> %f0, <8 x double> %f1, <8 x i1> %m, i32 %n)
  %r1 = call <8 x double> @llvm.vp.fsub.v8f64(<8 x double> %f0, <8 x double> %f1, <8 x i1> %m, i32 %n)
  %r2 = call <8 x double> @llvm.vp.fmul.v8f64(<8 x double> %f0, <8 x double> %f1, <8 x i1> %m, i32 %n)
  %r3 = call <8 x double> @llvm.vp.fdiv.v8f64(<8 x double> %f0, <8 x double> %f1, <8 x i1> %m, i32 %n)
  %r4 = call <8 x double> @llvm.vp.frem.v8f64(<8 x double> %f0, <8 x double> %f1, <8 x i1> %m, i32 %n)
  %r5 = call <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double> %f0, i32 639, <8 x i1> %m, i32 %n)
  ret void
}

; TODO: test_vp_constrained_fp


define void @test_vp_reduction(i32 %x, <8 x i32> %vi, <8 x float> %vf, float %f, <8 x i1> %m, i32 %n) {
  %r0 = call i32 @llvm.vp.reduce.add.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r1 = call i32 @llvm.vp.reduce.mul.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r2 = call i32 @llvm.vp.reduce.and.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r3 = call i32 @llvm.vp.reduce.or.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r4 = call i32 @llvm.vp.reduce.xor.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r5 = call i32 @llvm.vp.reduce.smax.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r6 = call i32 @llvm.vp.reduce.smin.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r7 = call i32 @llvm.vp.reduce.umax.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r8 = call i32 @llvm.vp.reduce.umin.v8i32(i32 %x, <8 x i32> %vi, <8 x i1> %m, i32 %n)
  %r9 = call float @llvm.vp.reduce.fmin.v8f32(float %f, <8 x float> %vf, <8 x i1> %m, i32 %n)
  %rA = call float @llvm.vp.reduce.fmax.v8f32(float %f, <8 x float> %vf, <8 x i1> %m, i32 %n)
  %rB = call float @llvm.vp.reduce.fminimum.v8f32(float %f, <8 x float> %vf, <8 x i1> %m, i32 %n)
  %rC = call float @llvm.vp.reduce.fmaximum.v8f32(float %f, <8 x float> %vf, <8 x i1> %m, i32 %n)
  %rD = call float @llvm.vp.reduce.fadd.v8f32(float %f, <8 x float> %vf, <8 x i1> %m, i32 %n)
  %rE = call float @llvm.vp.reduce.fmul.v8f32(float %f, <8 x float> %vf, <8 x i1> %m, i32 %n)
  ret void
}

define void @test_vp_splice0(<8 x i32> %i0, <8 x i32> %i1, <8 x i1> %m, i32 %l0, i32 %l1) {
  %r0 = call <8 x i32> @llvm.experimental.vp.splice.v8i32(<8 x i32> %i0, <8 x i32> %i1, i32 2, <8 x i1> %m, i32 %l0, i32 %l1)
  ret void
}

define void @test_vp_splice1(<vscale x 8 x i32> %i0, <vscale x 8 x i32> %i1, <vscale x 8 x i1> %m, i32 %l0, i32 %l1) {
  %r0 = call <vscale x 8 x i32> @llvm.experimental.vp.splice.nxv8i32(<vscale x 8 x i32> %i0, <vscale x 8 x i32> %i1, i32 -1, <vscale x 8 x i1> %m, i32 %l0, i32 %l1)
  ret void
}

define void @test_vp_conversions(<8 x ptr> %p0, <8 x i32> %i0, <8 x i64> %i1, <8 x float> %f0, <8 x double> %f1, <8 x i1> %mask, i32 %evl) {
  %r0 = call <8 x i32> @llvm.vp.fptoui.v8i32.v8f32(<8 x float> %f0, <8 x i1> %mask, i32 %evl)
  %r1 = call <8 x i32> @llvm.vp.fptosi.v8i32.v8f32(<8 x float> %f0, <8 x i1> %mask, i32 %evl)
  %r2 = call <8 x float> @llvm.vp.uitofp.v8f32.v8i32(<8 x i32> %i0, <8 x i1> %mask, i32 %evl)
  %r3 = call <8 x float> @llvm.vp.sitofp.v8f32.v8i32(<8 x i32> %i0, <8 x i1> %mask, i32 %evl)
  %r4 = call <8 x float> @llvm.vp.fptrunc.v8f32.v8f64(<8 x double> %f1, <8 x i1> %mask, i32 %evl)
  %r5 = call <8 x double> @llvm.vp.fpext.v8f64.v8f32(<8 x float> %f0, <8 x i1> %mask, i32 %evl)
  %r6 = call <8 x i32> @llvm.vp.trunc.v8i32.v8i64(<8 x i64> %i1, <8 x i1> %mask, i32 %evl)
  %r7 = call <8 x i64> @llvm.vp.zext.v8i64.v8i32(<8 x i32> %i0, <8 x i1> %mask, i32 %evl)
  %r8 = call <8 x i64> @llvm.vp.sext.v8i64.v8i32(<8 x i32> %i0, <8 x i1> %mask, i32 %evl)
  %r9 = call <8 x i32> @llvm.vp.ptrtoint.v8i32.v8p0(<8 x ptr> %p0, <8 x i1> %mask, i32 %evl)
  %r10 = call <8 x ptr> @llvm.vp.inttoptr.v8p0.v8i32(<8 x i32> %i0, <8 x i1> %mask, i32 %evl)
  ret void
}

define void @test_vp_comparisons(<8 x float> %f0, <8 x float> %f1, <8 x i32> %i0, <8 x i32> %i1, <8 x i1> %mask, i32 %evl) {
  %r0 = call <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float> %f0, <8 x float> %f1, metadata !"oeq", <8 x i1> %mask, i32 %evl)
  %r1 = call <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32> %i0, <8 x i32> %i1, metadata !"eq", <8 x i1> %mask, i32 %evl)
  ret void
}

; integer arith
declare <8 x i32> @llvm.vp.add.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sub.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.urem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
; bit arith
declare <8 x i32> @llvm.vp.and.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.or.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.xor.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.shl.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
; fp arith
declare <8 x double> @llvm.vp.fadd.v8f64(<8 x double>, <8 x double>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.fsub.v8f64(<8 x double>, <8 x double>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.fmul.v8f64(<8 x double>, <8 x double>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.fdiv.v8f64(<8 x double>, <8 x double>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.frem.v8f64(<8 x double>, <8 x double>, <8 x i1>, i32)
declare <8 x i1> @llvm.vp.is.fpclass.v8f64(<8 x double>, i32, <8 x i1>, i32)
; reductions
declare i32 @llvm.vp.reduce.add.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.mul.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.and.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.or.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.xor.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.smax.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.smin.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.umax.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare i32 @llvm.vp.reduce.umin.v8i32(i32, <8 x i32>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmin.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmax.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fminimum.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmaximum.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fadd.v8f32(float, <8 x float>, <8 x i1>, i32)
declare float @llvm.vp.reduce.fmul.v8f32(float, <8 x float>, <8 x i1>, i32)
; casts
declare <8 x i32> @llvm.vp.fptoui.v8i32.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.fptosi.v8i32.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.uitofp.v8f32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.sitofp.v8f32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fptrunc.v8f32.v8f64(<8 x double>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.fpext.v8f64.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.trunc.v8i32.v8i64(<8 x i64>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.zext.v8i64.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.sext.v8i64.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ptrtoint.v8i32.v8p0(<8 x ptr>, <8 x i1>, i32)
declare <8 x ptr> @llvm.vp.inttoptr.v8p0.v8i32(<8 x i32>, <8 x i1>, i32)
; compares
declare <8 x i1> @llvm.vp.fcmp.v8f32(<8 x float>, <8 x float>, metadata, <8 x i1>, i32)
declare <8 x i1> @llvm.vp.icmp.v8i32(<8 x i32>, <8 x i32>, metadata, <8 x i1>, i32)
; shuffles
declare <8 x i32> @llvm.experimental.vp.splice.v8i32(<8 x i32>, <8 x i32>, i32, <8 x i1>, i32, i32)
declare <vscale x 8 x i32> @llvm.experimental.vp.splice.nxv8i32(<vscale x 8 x i32>, <vscale x 8 x i32>, i32, <vscale x 8 x i1>, i32, i32)
