;;OpImageSampleExplicitLod_arg.cl
;;void __kernel sample_kernel_read( __global float4 *results,
;;				  read_only image2d_t image,
;;			          sampler_t imageSampler,
;;				  float2 coord,
;;				  float2 dx,
;;				  float2 dy)
;;{
;;   *results = read_imagef( image, imageSampler, coord);
;;   *results = read_imagef( image, imageSampler, coord, 3.14f);
;;   *results = read_imagef( image, imageSampler, coord, dx, dy);
;;}
;;clang -cc1 -O0 -triple spir-unknown-unknown -cl-std=CL2.0 -x cl OpImageSampleExplicitLod_arg.cl -include opencl-20.h -emit-llvm -o - | opt -mem2reg -S > OpImageSampleExplicitLod_arg2.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeFloat [[float:[0-9]+]] 32
; CHECK-SPIRV: Constant [[float]] [[lodNull:[0-9]+]] 0
; CHECK-SPIRV: Constant [[float]] [[lod:[0-9]+]] 1078523331
; CHECK-SPIRV: FunctionParameter
; CHECK-SPIRV: FunctionParameter
; CHECK-SPIRV: FunctionParameter
; CHECK-SPIRV: FunctionParameter
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[dx:[0-9]+]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[dy:[0-9]+]]

; CHECK-SPIRV: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 2 [[lodNull]]
; CHECK-SPIRV: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 2 [[lod]]
; CHECK-SPIRV: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 4 [[dx]] [[dy]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @sample_kernel_read(<4 x float> addrspace(1)* %results, %opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, <2 x float> %dx, <2 x float> %dy) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord)
; CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord)

  store <4 x float> %call, <4 x float> addrspace(1)* %results, align 16
  %call1 = call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, float 0x40091EB860000000)
; CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, float 0x40091EB860000000)

  store <4 x float> %call1, <4 x float> addrspace(1)* %results, align 16
  %call2 = call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_fDv2_fDv2_f(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, <2 x float> %dx, <2 x float> %dy)
; CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_fS_S_(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, <2 x float> %dx, <2 x float> %dy)

  store <4 x float> %call2, <4 x float> addrspace(1)* %results, align 16
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t*, <2 x float>) #1

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t*, <2 x float>, float) #1

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_fDv2_fDv2_f(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t*, <2 x float>, <2 x float>, <2 x float>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1, i32 0, i32 0, i32 0, i32 0}
!2 = !{!"none", !"read_only", !"none", !"none", !"none", !"none"}
!3 = !{!"float4*", !"image2d_t", !"sampler_t", !"float2", !"float2", !"float2"}
!4 = !{!"float4*", !"image2d_t", !"sampler_t", !"float2", !"float2", !"float2"}
!5 = !{!"", !"", !"", !"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"cl_images"}

