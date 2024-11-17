; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability StorageImageReadWithoutFormat
; CHECK-SPIRV: %[[#]] = OpImageRead %[[#]] %[[#]] %[[#]] Sample %[[#]]

define spir_kernel void @sample_test(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0) %source, i32 %sampler, <4 x float> addrspace(1)* nocapture %results) {
entry:
  %call = tail call spir_func i32 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z13get_global_idj(i32 1)
  %call2 = tail call spir_func i32 @_Z15get_image_width19ocl_image2d_msaa_ro(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0) %source)
  %call3 = tail call spir_func i32 @_Z16get_image_height19ocl_image2d_msaa_ro(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0) %source)
  %call4 = tail call spir_func i32 @_Z21get_image_num_samples19ocl_image2d_msaa_ro(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0) %source)
  %cmp20 = icmp eq i32 %call4, 0
  br i1 %cmp20, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %vecinit = insertelement <2 x i32> undef, i32 %call, i32 0
  %vecinit8 = insertelement <2 x i32> %vecinit, i32 %call1, i32 1
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %sample.021 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %mul5 = mul i32 %sample.021, %call3
  %tmp = add i32 %mul5, %call1
  %tmp19 = mul i32 %tmp, %call2
  %add7 = add i32 %tmp19, %call
  %call9 = tail call spir_func <4 x float> @_Z11read_imagef19ocl_image2d_msaa_roDv2_ii(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0) %source, <2 x i32> %vecinit8, i32 %sample.021)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %results, i32 %add7
  store <4 x float> %call9, <4 x float> addrspace(1)* %arrayidx, align 16
  %inc = add nuw i32 %sample.021, 1
  %cmp = icmp ult i32 %inc, %call4
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32)

declare spir_func i32 @_Z15get_image_width19ocl_image2d_msaa_ro(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0))

declare spir_func i32 @_Z16get_image_height19ocl_image2d_msaa_ro(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0))

declare spir_func i32 @_Z21get_image_num_samples19ocl_image2d_msaa_ro(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0))

declare spir_func <4 x float> @_Z11read_imagef19ocl_image2d_msaa_roDv2_ii(target("spirv.Image", void, 1, 0, 0, 1, 0, 0, 0), <2 x i32>, i32)
