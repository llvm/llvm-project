; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: %[[#RetID:]] = OpImageSampleExplicitLod %[[#RetType:]] %[[#]] %[[#]] Lod %[[#]]
; CHECK-SPIRV-DAG: %[[#RetType]] = OpTypeVector %[[#]] 4
; CHECK-SPIRV:     %[[#]] = OpCompositeExtract %[[#]] %[[#RetID]] 0

%opencl.image2d_depth_ro_t = type opaque

define spir_kernel void @sample_kernel(%opencl.image2d_depth_ro_t addrspace(1)* %input, i32 %imageSampler, float addrspace(1)* %xOffsets, float addrspace(1)* %yOffsets, float addrspace(1)* %results) {
entry:
  %call = call spir_func i32 @_Z13get_global_idj(i32 0)
  %call1 = call spir_func i32 @_Z13get_global_idj(i32 1)
  %call2.tmp1 = call spir_func <2 x i32> @_Z13get_image_dim20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)* %input)
  %call2.old = extractelement <2 x i32> %call2.tmp1, i32 0
  %mul = mul i32 %call1, %call2.old
  %add = add i32 %mul, %call
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %xOffsets, i32 %add
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %conv = fptosi float %0 to i32
  %vecinit = insertelement <2 x i32> undef, i32 %conv, i32 0
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %yOffsets, i32 %add
  %1 = load float, float addrspace(1)* %arrayidx3, align 4
  %conv4 = fptosi float %1 to i32
  %vecinit5 = insertelement <2 x i32> %vecinit, i32 %conv4, i32 1
  %call6.tmp.tmp = call spir_func float @_Z11read_imagef20ocl_image2d_depth_ro11ocl_samplerDv2_i(%opencl.image2d_depth_ro_t addrspace(1)* %input, i32 %imageSampler, <2 x i32> %vecinit5)
  %arrayidx7 = getelementptr inbounds float, float addrspace(1)* %results, i32 %add
  store float %call6.tmp.tmp, float addrspace(1)* %arrayidx7, align 4
  ret void
}

declare spir_func float @_Z11read_imagef20ocl_image2d_depth_ro11ocl_samplerDv2_i(%opencl.image2d_depth_ro_t addrspace(1)*, i32, <2 x i32>)

declare spir_func i32 @_Z13get_global_idj(i32)

declare spir_func <2 x i32> @_Z13get_image_dim20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)*)
