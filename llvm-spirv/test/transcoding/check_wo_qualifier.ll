; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r -spirv-gen-image-type-acc-postfix %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: opencl.image2d_array_wo_t = type opaque
; CHECK-LLVM: define spir_kernel void @sample_kernel(%opencl.image2d_array_wo_t addrspace(1)
; CHECK-LLVM-SAME: !kernel_arg_access_qual [[AQ:![0-9]+]]
; CHECK-LLVM-SAME: !kernel_arg_type [[TYPE:![0-9]+]]
; CHECK-LLVM-SAME: !kernel_arg_base_type [[TYPE]]

; CHECK-LLVM: call spir_func <2 x i32> @_Z13get_image_dimPU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)
; CHECK-LLVM: call spir_func i64 @_Z20get_image_array_sizePU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)
; CHECK-LLVM: declare spir_func <2 x i32> @_Z13get_image_dimPU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)
; CHECK-LLVM: declare spir_func i64 @_Z20get_image_array_sizePU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)
; CHECK-LLVM-DAG: [[AQ]] = !{!"write_only"}
; CHECK-LLVM-DAG: [[TYPE]] = !{!"image2d_array_t"}

; ModuleID = 'out.ll'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%opencl.image2d_array_wo_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @sample_kernel(%opencl.image2d_array_wo_t addrspace(1)* %input) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
entry:
  %call.tmp1 = call spir_func <2 x i32> @_Z13get_image_dimPU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)* %input)
  %call.tmp2 = shufflevector <2 x i32> %call.tmp1, <2 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %call.tmp3 = call spir_func i64 @_Z20get_image_array_sizePU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)* %input)
  %call.tmp4 = trunc i64 %call.tmp3 to i32
  %call.tmp5 = insertelement <3 x i32> %call.tmp2, i32 %call.tmp4, i32 2
  %call.old = extractelement <3 x i32> %call.tmp5, i32 0
  ret void
}

; Function Attrs: nounwind
declare spir_func <2 x i32> @_Z13get_image_dimPU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)*) #0

; Function Attrs: nounwind
declare spir_func i64 @_Z20get_image_array_sizePU3AS125opencl.image2d_array_wo_t(%opencl.image2d_array_wo_t addrspace(1)*) #0

attributes #0 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!6}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!spirv.Generator = !{!10}

!1 = !{i32 1}
!2 = !{!"write_only"}
!3 = !{!"image2d_array_t"}
!4 = !{!""}
!5 = !{!"image2d_array_t"}
!6 = !{i32 3, i32 102000}
!7 = !{i32 1, i32 2}
!8 = !{}
!9 = !{!"cl_images"}
!10 = !{i16 6, i16 14}
