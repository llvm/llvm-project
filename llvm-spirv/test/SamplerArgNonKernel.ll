; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

;CHECK: EntryPoint {{[0-9]+}} [[KernelId:[0-9]+]]

%opencl.image2d_t = type opaque
;CHECK: TypeImage [[image2d_t:[0-9]+]]
;CHECK: TypeSampler [[sampler_t:[0-9]+]]
;CHECK: TypeSampledImage [[sampled_image_t:[0-9]+]]

; Function Attrs: nounwind
define spir_func float @test(%opencl.image2d_t addrspace(1)* %Img, i32 %Smp) #0 {
;CHECK-NOT: Function {{[0-9]+}} [[KernelId]]
;CHECK: Function
;CHECK: FunctionParameter [[image2d_t]] [[image:[0-9]+]]
;CHECK: FunctionParameter [[sampler_t]] [[sampler:[0-9]+]]
entry:
  %call = call spir_func <4 x i32> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(%opencl.image2d_t addrspace(1)* %Img, i32 %Smp, <2 x i32> zeroinitializer)
;CHECK: SampledImage [[sampled_image_t]] [[sampled_image:[0-9]+]] [[image]] [[sampler]]
;CHECK: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} [[sampled_image]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}

  %0 = extractelement <4 x i32> %call, i32 0
  %conv = sitofp i32 %0 to float
  ret float %conv
}

declare spir_func <4 x i32> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(%opencl.image2d_t addrspace(1)*, i32, <2 x i32>) #1

; Function Attrs: nounwind
define spir_kernel void @test2(%opencl.image2d_t addrspace(1)* %Img, float addrspace(1)* %result) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
;CHECK: Function  {{[0-9]+}} [[KernelId]]
entry:
  %call = call spir_func float @test(%opencl.image2d_t addrspace(1)* %Img, i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %result, i32 0
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %add = fadd float %0, %call
  store float %add, float addrspace(1)* %arrayidx, align 4
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1}
!2 = !{!"read_only", !"none"}
!3 = !{!"image2d_t", !"float*"}
!4 = !{!"image2d_t", !"float*"}
!5 = !{!"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"cl_images"}

