; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs -amdgpu-enable-hot-block-remat -amdgpu-remat-enable-sub-exp-remat

; Regression test for PHI being sinked to uses as a pacifist.
; Just checking that the test does not crash.

; ModuleID = 'reduced.ll'
source_filename = "reduced.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn--amdpal"

define amdgpu_ps void @_amdgpu_ps_main(float %arg, float %arg1, float %arg2, float %arg3, float %arg4, i32 %arg5, float %arg6, float %arg7, float %arg8, <2 x half> %arg9, i1 %arg10) #0 {
bb:
  br label %bb19

bb11:                                             ; preds = %bb19
  %i = bitcast i32 %i21 to float
  %i12 = bitcast i32 %i23 to float
  %i13 = fmul float 0.000000e+00, %i26
  %i14 = fmul float %i13, 0.000000e+00
  %i15 = fmul float %i12, %i
  %i16 = fadd float %i15, %i14
  %i17 = select i1 false, float 0.000000e+00, float %i16
  %i18 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %arg4, float %arg8)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 0, <2 x half> %i18, <2 x half> %arg9, i1 false, i1 false)
  ret void

bb19:                                             ; preds = %bb19, %bb
  %i20 = phi i32 [ 0, %bb19 ], [ %arg5, %bb ]
  %i21 = phi i32 [ %i35, %bb19 ], [ 0, %bb ]
  %i22 = phi i32 [ %i38, %bb19 ], [ 0, %bb ]
  %i23 = phi i32 [ %i60, %bb19 ], [ 0, %bb ]
  %i24 = phi i32 [ %i61, %bb19 ], [ 0, %bb ]
  %i25 = phi i32 [ %i62, %bb19 ], [ 0, %bb ]
  %i26 = phi float [ %i39, %bb19 ], [ 0.000000e+00, %bb ]
  %i27 = phi i32 [ %i49, %bb19 ], [ 0, %bb ]
  %i28 = phi i32 [ %i50, %bb19 ], [ 0, %bb ]
  %i29 = phi i32 [ %i51, %bb19 ], [ 0, %bb ]
  %i30 = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32.v8i32(i32 1, i32 %i20, i32 0, <8 x i32> zeroinitializer, i32 0, i32 0)
  %i31 = extractelement <4 x float> %i30, i64 0
  %i32 = fmul float %arg1, %i31
  %i33 = bitcast i32 %i22 to float
  %i34 = fmul float %arg, %i32
  %i35 = select i1 %arg10, i32 %arg5, i32 %i21
  %i36 = fadd float 0.000000e+00, %i33
  %i37 = bitcast float %i36 to i32
  %i38 = select i1 %arg10, i32 %i22, i32 %i37
  %i39 = fadd float %i26, 1.000000e+00
  %i40 = bitcast i32 %i27 to float
  %i41 = bitcast i32 %i28 to float
  %i42 = bitcast i32 %i29 to float
  %i43 = fadd float 0.000000e+00, %i40
  %i44 = fadd float 0.000000e+00, %i41
  %i45 = fadd float 0.000000e+00, %i42
  %i46 = bitcast float %i43 to i32
  %i47 = bitcast float %i44 to i32
  %i48 = bitcast float %i45 to i32
  %i49 = select i1 %arg10, i32 %i27, i32 %i46
  %i50 = select i1 %arg10, i32 %i28, i32 %i47
  %i51 = select i1 %arg10, i32 %i29, i32 %i48
  %i52 = fmul float %i34, %arg7
  %i53 = bitcast i32 %i24 to float
  %i54 = bitcast i32 %i25 to float
  %i55 = fadd float %arg6, %i53
  %i56 = fadd float %arg2, %i54
  %i57 = bitcast float %i52 to i32
  %i58 = bitcast float %i55 to i32
  %i59 = bitcast float %i56 to i32
  %i60 = select i1 %arg10, i32 %i57, i32 %i23
  %i61 = select i1 %arg10, i32 %i58, i32 %i24
  %i62 = select i1 %arg10, i32 %i59, i32 %i25
  %i63 = sitofp i32 %i20 to float
  %i64 = fcmp olt float %arg3, %i63
  br i1 %i64, label %bb11, label %bb19
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.amdgcn.exp.compr.v2f16(i32 immarg, i32 immarg, <2 x half>, <2 x half>, i1 immarg, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32.v8i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #3

attributes #0 = { "target-features"=",+wavefrontsize64,+cumode,-xnack" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(read) }
