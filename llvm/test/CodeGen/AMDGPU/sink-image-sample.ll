; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; Test that image.sample instruction is sunk across the branch and not left in the first block. Since the kill may terminate the shader there might be no need for sampling the image.

; GCN-LABEL: {{^}}sinking_img_sample:
; GCN-NOT: image_sample
; GCN: branch
; GCN: image_sample
; GCN: exp null

define amdgpu_ps float @sinking_img_sample() {
main_body:
  %i = call <3 x float> @llvm.amdgcn.image.sample.2d.v3f32.f32(i32 7, float undef, float undef, <8 x i32> undef, <4 x i32> undef, i1 false, i32 0, i32 0)
  br i1 undef, label %endif1, label %if1

if1:                                              ; preds = %main_body
  call void @llvm.amdgcn.kill(i1 false) #4
  br label %exit

endif1:                                           ; preds = %main_body
  %i22 = extractelement <3 x float> %i, i32 2
  %i23 = call nsz arcp contract float @llvm.fma.f32(float %i22, float 0.000000e+00, float 0.000000e+00) #1
  br label %exit

exit:                                             ; preds = %endif1, %if1
  %i24 = phi float [ undef, %if1 ], [ %i23, %endif1 ]
  ret float %i24
}
; Function Attrs: nounwind readonly willreturn
declare <3 x float> @llvm.amdgcn.image.sample.2d.v3f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #3

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fma.f32(float, float, float) #2

; Function Attrs: nounwind
declare void @llvm.amdgcn.kill(i1) #4

attributes #1 = { nounwind readnone }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nounwind readonly willreturn }
attributes #4 = { nounwind }
