; Source:
; constant sampler_t constSampl = CLK_FILTER_LINEAR;
;
; __kernel
; void sample_kernel(image2d_t input, float2 coords, global float4 *results, sampler_t argSampl) {
;   *results = read_imagef(input, constSampl, coords);
;   *results = read_imagef(input, argSampl, coords);
;   *results = read_imagef(input, CLK_FILTER_NEAREST|CLK_ADDRESS_REPEAT, coords);
; }
;clang -cc1 -triple spir -cl-std=CL2.0 sampler.cl -finclude-default-header -emit-llvm -o -

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability LiteralSampler
; CHECK-SPIRV: EntryPoint 6 [[sample_kernel:[0-9]+]] "sample_kernel"

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque

; CHECK-SPIRV: TypeSampler [[TypeSampler:[0-9]+]]
; CHECK-SPIRV: TypeSampledImage [[SampledImageTy:[0-9]+]]
; CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler1:[0-9]+]] 0 0 1
; CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler2:[0-9]+]] 3 0 0


; Function Attrs: convergent nounwind
; CHECK-SPIRV: Function {{.*}} [[sample_kernel]]
; CHECK-SPIRV: FunctionParameter {{.*}} [[InputImage:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TypeSampler]] [[argSampl:[0-9]+]]
; CHECK-LLVM: define spir_kernel void @sample_kernel(%opencl.image2d_ro_t addrspace(1)* %input, <2 x float> %coords, <4 x float> addrspace(1)* nocapture %results, %opencl.sampler_t* %argSampl)
define spir_kernel void @sample_kernel(%opencl.image2d_ro_t addrspace(1)* %input, <2 x float> %coords, <4 x float> addrspace(1)* nocapture %results, %opencl.sampler_t addrspace(2)* %argSampl) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
entry:
  %0 = tail call %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 32) #2

; CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage1:[0-9]+]] [[InputImage]] [[ConstSampler1]]
; CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage1]]
; CHECK-LLVM:  call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t* %0, <2 x float> %coords)
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %0, <2 x float> %coords) #3
  store <4 x float> %call, <4 x float> addrspace(1)* %results, align 16, !tbaa !12

; CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage2:[0-9]+]] [[InputImage]] [[argSampl]]
; CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage2]]
; CHECK-LLVM:   call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t* %argSampl, <2 x float> %coords)
  %call1 = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %argSampl, <2 x float> %coords) #3
  store <4 x float> %call1, <4 x float> addrspace(1)* %results, align 16, !tbaa !12
  %1 = tail call %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 22) #2

; CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage3:[0-9]+]] [[InputImage]] [[ConstSampler2]]
; CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage3]]
; CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t* %{{[0-9]+}}, <2 x float> %coords)
  %call2 = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %1, <2 x float> %coords) #3
  store <4 x float> %call2, <4 x float> addrspace(1)* %results, align 16, !tbaa !12
  ret void
}

; Function Attrs: convergent nounwind readonly
; CHECK-LLVM: declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t*, <2 x float>)
declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, <2 x float>) local_unnamed_addr #1

declare %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32) local_unnamed_addr

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { convergent nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_images"}
!4 = !{!"clang version 6.0.0 (cfe/trunk)"}
!5 = !{i32 1, i32 0, i32 1, i32 0}
!6 = !{!"read_only", !"none", !"none", !"none"}
!7 = !{!"image2d_t", !"float2", !"float4*", !"sampler_t"}
!8 = !{!"image2d_t", !"float __attribute__((ext_vector_type(2)))", !"float __attribute__((ext_vector_type(4)))*", !"sampler_t"}
!9 = !{!"", !"", !"", !""}
!12 = !{!13, !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
