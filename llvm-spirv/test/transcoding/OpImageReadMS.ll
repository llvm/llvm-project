; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef19ocl_image2d_msaa_roDv2_ii(%opencl.image2d_msaa_ro_t

; CHECK-SPIRV: 7 ImageRead {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 64 {{[0-9]+}}

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_msaa_ro_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @sample_test(%opencl.image2d_msaa_ro_t addrspace(1)* %source, i32 %sampler, <4 x float> addrspace(1)* nocapture %results) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = tail call spir_func i32 @_Z13get_global_idj(i32 0) #2
  %call1 = tail call spir_func i32 @_Z13get_global_idj(i32 1) #2
  %call2 = tail call spir_func i32 @_Z15get_image_width19ocl_image2d_msaa_ro(%opencl.image2d_msaa_ro_t addrspace(1)* %source) #2
  %call3 = tail call spir_func i32 @_Z16get_image_height19ocl_image2d_msaa_ro(%opencl.image2d_msaa_ro_t addrspace(1)* %source) #2
  %call4 = tail call spir_func i32 @_Z21get_image_num_samples19ocl_image2d_msaa_ro(%opencl.image2d_msaa_ro_t addrspace(1)* %source) #2
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
  %call9 = tail call spir_func <4 x float> @_Z11read_imagef19ocl_image2d_msaa_roDv2_ii(%opencl.image2d_msaa_ro_t addrspace(1)* %source, <2 x i32> %vecinit8, i32 %sample.021) #2
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

declare spir_func i32 @_Z13get_global_idj(i32) #1

declare spir_func i32 @_Z15get_image_width19ocl_image2d_msaa_ro(%opencl.image2d_msaa_ro_t addrspace(1)*) #1

declare spir_func i32 @_Z16get_image_height19ocl_image2d_msaa_ro(%opencl.image2d_msaa_ro_t addrspace(1)*) #1

declare spir_func i32 @_Z21get_image_num_samples19ocl_image2d_msaa_ro(%opencl.image2d_msaa_ro_t addrspace(1)*) #1

declare spir_func <4 x float> @_Z11read_imagef19ocl_image2d_msaa_roDv2_ii(%opencl.image2d_msaa_ro_t addrspace(1)*, <2 x i32>, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!10}

!1 = !{i32 1, i32 0, i32 1}
!2 = !{!"read_only", !"none", !"none"}
!3 = !{!"image2d_msaa_t", !"sampler_t", !"float4*"}
!4 = !{!"image2d_msaa_t", !"sampler_t", !"float4*"}
!5 = !{!"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"cl_khr_gl_msaa_sharing"}
!9 = !{!"cl_images"}
!10 = !{}
