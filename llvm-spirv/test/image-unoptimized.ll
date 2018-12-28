; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_t = type opaque

; Function Attrs: nounwind
; CHECK: {{[0-9]*}} Store
; CHECK-NEXT: 1 Return
define spir_kernel void @test_fn(%opencl.image2d_t addrspace(1)* %srcimg, i32 %sampler, <4 x float> addrspace(1)* %results) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %srcimg.addr = alloca %opencl.image2d_t addrspace(1)*, align 4
  %sampler.addr = alloca i32, align 4
  %results.addr = alloca <4 x float> addrspace(1)*, align 4
  %tid_x = alloca i32, align 4
  %tid_y = alloca i32, align 4
  %.compoundliteral = alloca <2 x i32>, align 8
  store %opencl.image2d_t addrspace(1)* %srcimg, %opencl.image2d_t addrspace(1)** %srcimg.addr, align 4
  store i32 %sampler, i32* %sampler.addr, align 4
  store <4 x float> addrspace(1)* %results, <4 x float> addrspace(1)** %results.addr, align 4
  %call = call spir_func i32 @_Z13get_global_idj(i32 0) #2
  store i32 %call, i32* %tid_x, align 4
  %call1 = call spir_func i32 @_Z13get_global_idj(i32 1) #2
  store i32 %call1, i32* %tid_y, align 4
  %0 = load %opencl.image2d_t addrspace(1)*, %opencl.image2d_t addrspace(1)** %srcimg.addr, align 4
  %1 = load i32, i32* %sampler.addr, align 4
  %2 = load i32, i32* %tid_x, align 4
  %vecinit = insertelement <2 x i32> undef, i32 %2, i32 0
  %3 = load i32, i32* %tid_y, align 4
  %vecinit2 = insertelement <2 x i32> %vecinit, i32 %3, i32 1
  store <2 x i32> %vecinit2, <2 x i32>* %.compoundliteral
  %4 = load <2 x i32>, <2 x i32>* %.compoundliteral
  %call3 = call spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(%opencl.image2d_t addrspace(1)* %0, i32 %1, <2 x i32> %4) #2
  %5 = load i32, i32* %tid_y, align 4
  %6 = load %opencl.image2d_t addrspace(1)*, %opencl.image2d_t addrspace(1)** %srcimg.addr, align 4
  %call4 = call spir_func i32 @_Z15get_image_width11ocl_image2d(%opencl.image2d_t addrspace(1)* %6) #2
  %mul = mul nsw i32 %5, %call4
  %7 = load i32, i32* %tid_x, align 4
  %add = add nsw i32 %mul, %7
  %8 = load <4 x float> addrspace(1)*, <4 x float> addrspace(1)** %results.addr, align 4
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %8, i32 %add
  store <4 x float> %call3, <4 x float> addrspace(1)* %arrayidx, align 16
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(%opencl.image2d_t addrspace(1)*, i32, <2 x i32>) #1

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z15get_image_width11ocl_image2d(%opencl.image2d_t addrspace(1)*) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!7}

!1 = !{i32 1, i32 0, i32 1}
!2 = !{!"read_only", !"none", !"none"}
!3 = !{!"image2d_t", !"sampler_t", !"float4*"}
!4 = !{!"image2d_t", !"sampler_t", !"float4*"}
!5 = !{!"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{}
!8 = !{!"cl_images"}
