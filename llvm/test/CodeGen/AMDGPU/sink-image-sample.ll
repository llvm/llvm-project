; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; Test that image.sample LOD(_L), Level 0(_LZ), Derivative(_D) instructions are sunk across the branch and not left in the first block. Since the kill may terminate the shader there might be no need for sampling the image.

; GCN-LABEL: {{^}}sinking_img_sample:
; GCN-NOT: image_sample_l v
; GCN-NOT: image_sample_lz v
; GCN-NOT: image_sample_c_lz v
; GCN-NOT: image_sample_c_l v
; GCN-NOT: image_sample_d v
; GCN-NOT: image_sample_c_d v
; GCN-NOT: image_sample_d_cl v
; GCN-NOT: image_sample_c_d_cl v
; GCN: branch
; GCN: image_sample_l v
; GCN: image_sample_lz v
; GCN: image_sample_c_lz v
; GCN: image_sample_c_l v
; GCN: image_sample_d v
; GCN: image_sample_c_d v
; GCN: image_sample_d_cl v
; GCN: image_sample_c_d_cl v
; GCN: exp null

define amdgpu_ps float @sinking_img_sample(i1 %cond) {
main_body:
  %i1 = call <3 x float> @llvm.amdgcn.image.sample.l.2d.v3f32.f32(i32 7, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i2 = call <3 x float> @llvm.amdgcn.image.sample.lz.2d.v3f32.f32(i32 7, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i3 = call <3 x float> @llvm.amdgcn.image.sample.c.lz.2d.v3f32.f32(i32 7, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i4 = call <3 x float> @llvm.amdgcn.image.sample.c.l.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i5 = call <3 x float> @llvm.amdgcn.image.sample.d.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i6 = call <3 x float> @llvm.amdgcn.image.sample.c.d.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i7 = call <3 x float> @llvm.amdgcn.image.sample.d.cl.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i8 = call <3 x float> @llvm.amdgcn.image.sample.c.d.cl.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  br i1 %cond, label %endif1, label %if1

if1:                                              ; preds = %main_body
  call void @llvm.amdgcn.kill(i1 false) #4
  br label %exit

endif1:                                           ; preds = %main_body
  %i22 = extractelement <3 x float> %i1, i32 1
  %i23 = call nsz arcp contract float @llvm.fma.f32(float %i22, float 0.000000e+00, float 0.000000e+00) #1
  %i24 = extractelement <3 x float> %i2, i32 1
  %i25 = call nsz arcp contract float @llvm.fma.f32(float %i23, float %i24, float 0.000000e+00) #1
  %i26 = extractelement <3 x float> %i3, i32 1
  %i27 = call nsz arcp contract float @llvm.fma.f32(float %i25, float %i26, float 0.000000e+00) #1
  %i28 = extractelement <3 x float> %i4, i32 1
  %i29 = call nsz arcp contract float @llvm.fma.f32(float %i27, float %i28, float 0.000000e+00) #1
  %i30 = extractelement <3 x float> %i5, i32 1
  %i31 = call nsz arcp contract float @llvm.fma.f32(float %i29, float %i30, float 0.000000e+00) #1
  %i32 = extractelement <3 x float> %i6, i32 1
  %i33 = call nsz arcp contract float @llvm.fma.f32(float %i31, float %i32, float 0.000000e+00) #1
  %i34 = extractelement <3 x float> %i7, i32 1
  %i35 = call nsz arcp contract float @llvm.fma.f32(float %i33, float %i34, float 0.000000e+00) #1
  %i36 = extractelement <3 x float> %i8, i32 1
  %i37 = call nsz arcp contract float @llvm.fma.f32(float %i35, float %i36, float 0.000000e+00) #1
  br label %exit

exit:                                             ; preds = %endif1, %if1
  %i38 = phi float [ poison, %if1 ], [ %i37, %endif1 ]
  ret float %i38
}


; Test that image.sample instructions which use WQM are marked as Convergent and will be left in the first block.

; GCN-LABEL: {{^}}no_sinking_img_sample:
; GCN: image_sample v
; GCN: image_sample_c v
; GCN: image_sample_cl v
; GCN: image_sample_c_cl v
; GCN: image_sample_b v
; GCN: image_sample_c_b v
; GCN: image_sample_b_cl v
; GCN: branch
; GCN-NOT: image_sample v
; GCN-NOT: image_sample_c v
; GCN-NOT: image_sample_cl v
; GCN-NOT: image_sample_c_cl v
; GCN-NOT: image_sample_b v
; GCN-NOT: image_sample_c_b v
; GCN-NOT: image_sample_b_cl v
; GCN: exp null

define amdgpu_ps float @no_sinking_img_sample(i1 %cond) {
main_body:
  %i1 = call <3 x float> @llvm.amdgcn.image.sample.2d.v3f32.f32(i32 7, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i2 = call <3 x float> @llvm.amdgcn.image.sample.c.2d.v3f32.f32(i32 7, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i3 = call <3 x float> @llvm.amdgcn.image.sample.cl.2d.v3f32.f32(i32 7, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i4 = call <3 x float> @llvm.amdgcn.image.sample.c.cl.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i5 = call <3 x float> @llvm.amdgcn.image.sample.b.2d.v3f32.f32(i32 7, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i6 = call <3 x float> @llvm.amdgcn.image.sample.c.b.2d.v3f32.f32(i32 7, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i7 = call <3 x float> @llvm.amdgcn.image.sample.b.cl.2d.v3f32.f32.f32(i32 7, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  br i1 %cond, label %endif1, label %if1

if1:                                              ; preds = %main_body
  call void @llvm.amdgcn.kill(i1 false) #4
  br label %exit

endif1:                                           ; preds = %main_body
  %i22 = extractelement <3 x float> %i1, i32 2
  %i23 = call nsz arcp contract float @llvm.fma.f32(float %i22, float 0.000000e+00, float 0.000000e+00) #1
  %i24 = extractelement <3 x float> %i2, i32 2
  %i25 = call nsz arcp contract float @llvm.fma.f32(float %i23, float %i24, float 0.000000e+00) #1
  %i26 = extractelement <3 x float> %i3, i32 2
  %i27 = call nsz arcp contract float @llvm.fma.f32(float %i25, float %i26, float 0.000000e+00) #1
  %i28 = extractelement <3 x float> %i4, i32 2
  %i29 = call nsz arcp contract float @llvm.fma.f32(float %i27, float %i28, float 0.000000e+00) #1
  %i30 = extractelement <3 x float> %i5, i32 2
  %i31 = call nsz arcp contract float @llvm.fma.f32(float %i29, float %i30, float 0.000000e+00) #1
  %i32 = extractelement <3 x float> %i6, i32 2
  %i33 = call nsz arcp contract float @llvm.fma.f32(float %i31, float %i32, float 0.000000e+00) #1
  %i34 = extractelement <3 x float> %i7, i32 2
  %i35 = call nsz arcp contract float @llvm.fma.f32(float %i33, float %i34, float 0.000000e+00) #1
  br label %exit

exit:                                             ; preds = %endif1, %if1
  %i36 = phi float [ poison, %if1 ], [ %i35, %endif1 ]
  ret float %i36
}

; Function Attrs: nounwind readonly willreturn
declare <3 x float> @llvm.amdgcn.image.sample.2d.v3f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.2d.v3f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.cl.2d.v3f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.cl.2d.v3f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.b.2d.v3f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.b.2d.v3f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.b.cl.2d.v3f32.f32.(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.b.cl.2d.v3f32.f32(i32 immarg, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.l.2d.v3f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.lz.2d.v3f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.lz.2d.v3f32.f32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.l.2d.v3f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.d.2d.v3f32.f32.f32(i32 immarg, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.d.2d.v3f32.f32.f32(i32 immarg, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.d.cl.2d.v3f32.f32.f32(i32 immarg, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3
declare <3 x float> @llvm.amdgcn.image.sample.c.d.cl.2d.v3f32.f32.f32(i32 immarg, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fma.f32(float, float, float) #2

; Function Attrs: nounwind
declare void @llvm.amdgcn.kill(i1) #4

attributes #1 = { nounwind readnone }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nounwind readonly willreturn }
attributes #4 = { nounwind }
