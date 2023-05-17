; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK:     %[[#TypeImage:]] = OpTypeImage
; CHECK:     %[[#TypeSampler:]] = OpTypeSampler
; CHECK-DAG: %[[#TypeImagePtr:]] = OpTypePointer {{.*}} %[[#TypeImage]]
; CHECK-DAG: %[[#TypeSamplerPtr:]] = OpTypePointer {{.*}} %[[#TypeSampler]]

; CHECK:     %[[#srcimg:]] = OpFunctionParameter %[[#TypeImage]]
; CHECK:     %[[#sampler:]] = OpFunctionParameter %[[#TypeSampler]]

; CHECK:     %[[#srcimg_addr:]] = OpVariable %[[#TypeImagePtr]]
; CHECK:     %[[#sampler_addr:]] = OpVariable %[[#TypeSamplerPtr]]

; CHECK:     OpStore %[[#srcimg_addr]] %[[#srcimg]]
; CHECK:     OpStore %[[#sampler_addr]] %[[#sampler]]

; CHECK:     %[[#srcimg_val:]] = OpLoad %[[#]] %[[#srcimg_addr]]
; CHECK:     %[[#sampler_val:]] = OpLoad %[[#]] %[[#sampler_addr]]

; CHECK:     %[[#]] = OpSampledImage %[[#]] %[[#srcimg_val]] %[[#sampler_val]]
; CHECK-NEXT: OpImageSampleExplicitLod

; CHECK:     %[[#srcimg_val:]] = OpLoad %[[#]] %[[#srcimg_addr]]
; CHECK:     %[[#]] = OpImageQuerySizeLod %[[#]] %[[#srcimg_val]]

;; Excerpt from opencl-c-base.h
;; typedef float float4 __attribute__((ext_vector_type(4)));
;; typedef int int2 __attribute__((ext_vector_type(2)));
;; typedef __SIZE_TYPE__ size_t;
;;
;; Excerpt from opencl-c.h to speed up compilation.
;; #define __ovld __attribute__((overloadable))
;; #define __purefn __attribute__((pure))
;; #define __cnfn __attribute__((const))
;; size_t __ovld __cnfn get_global_id(unsigned int dimindx);
;; int __ovld __cnfn get_image_width(read_only image2d_t image);
;; float4 __purefn __ovld read_imagef(read_only image2d_t image, sampler_t sampler, int2 coord);
;;
;;
;; __kernel void test_fn(image2d_t srcimg, sampler_t sampler, global float4 *results) {
;;   int tid_x = get_global_id(0);
;;   int tid_y = get_global_id(1);
;;   results[tid_x + tid_y * get_image_width(srcimg)] = read_imagef(srcimg, sampler, (int2){tid_x, tid_y});
;; }

%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque

define dso_local spir_kernel void @test_fn(%opencl.image2d_ro_t addrspace(1)* %srcimg, %opencl.sampler_t addrspace(2)* %sampler, <4 x float> addrspace(1)* noundef %results) {
entry:
  %srcimg.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 4
  %sampler.addr = alloca %opencl.sampler_t addrspace(2)*, align 4
  %results.addr = alloca <4 x float> addrspace(1)*, align 4
  %tid_x = alloca i32, align 4
  %tid_y = alloca i32, align 4
  %.compoundliteral = alloca <2 x i32>, align 8
  store %opencl.image2d_ro_t addrspace(1)* %srcimg, %opencl.image2d_ro_t addrspace(1)** %srcimg.addr, align 4
  store %opencl.sampler_t addrspace(2)* %sampler, %opencl.sampler_t addrspace(2)** %sampler.addr, align 4
  store <4 x float> addrspace(1)* %results, <4 x float> addrspace(1)** %results.addr, align 4
  %call = call spir_func i32 @_Z13get_global_idj(i32 noundef 0)
  store i32 %call, i32* %tid_x, align 4
  %call1 = call spir_func i32 @_Z13get_global_idj(i32 noundef 1)
  store i32 %call1, i32* %tid_y, align 4
  %0 = load %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)** %srcimg.addr, align 4
  %1 = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** %sampler.addr, align 4
  %2 = load i32, i32* %tid_x, align 4
  %vecinit = insertelement <2 x i32> undef, i32 %2, i32 0
  %3 = load i32, i32* %tid_y, align 4
  %vecinit2 = insertelement <2 x i32> %vecinit, i32 %3, i32 1
  store <2 x i32> %vecinit2, <2 x i32>* %.compoundliteral, align 8
  %4 = load <2 x i32>, <2 x i32>* %.compoundliteral, align 8
  %call3 = call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_i(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.sampler_t addrspace(2)* %1, <2 x i32> noundef %4)
  %5 = load <4 x float> addrspace(1)*, <4 x float> addrspace(1)** %results.addr, align 4
  %6 = load i32, i32* %tid_x, align 4
  %7 = load i32, i32* %tid_y, align 4
  %8 = load %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)** %srcimg.addr, align 4
  %call4 = call spir_func i32 @_Z15get_image_width14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)* %8)
  %mul = mul nsw i32 %7, %call4
  %add = add nsw i32 %6, %mul
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %5, i32 %add
  store <4 x float> %call3, <4 x float> addrspace(1)* %arrayidx, align 16
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32 noundef)

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_i(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, <2 x i32> noundef)

declare spir_func i32 @_Z15get_image_width14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)*)
