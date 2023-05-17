; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability ImageReadWrite
; CHECK-SPIRV-DAG: OpCapability LiteralSampler
; CHECK-SPIRV-DAG: %[[#TyVoid:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#TyImageID:]] = OpTypeImage %[[#TyVoid]] 1D 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV-DAG: %[[#TySampledImageID:]] = OpTypeSampledImage %[[#TyImageID]]

; CHECK-SPIRV-DAG: %[[#ResID:]] = OpSampledImage %[[#TySampledImageID]]
; CHECK-SPIRV:     %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#ResID]]

%opencl.image1d_rw_t = type opaque

define spir_func void @sampFun(%opencl.image1d_rw_t addrspace(1)* %image) {
entry:
  %image.addr = alloca %opencl.image1d_rw_t addrspace(1)*, align 4
  store %opencl.image1d_rw_t addrspace(1)* %image, %opencl.image1d_rw_t addrspace(1)** %image.addr, align 4
  %0 = load %opencl.image1d_rw_t addrspace(1)*, %opencl.image1d_rw_t addrspace(1)** %image.addr, align 4
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(%opencl.image1d_rw_t addrspace(1)* %0, i32 8, i32 2)
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(%opencl.image1d_rw_t addrspace(1)*, i32, i32)
