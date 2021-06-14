; RUN: opt --verify %s

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
  ret void
}

; TODO: test_vp_constrained_fp

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
