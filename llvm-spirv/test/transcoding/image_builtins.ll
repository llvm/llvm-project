; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ModuleID = 'image_builtins.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque
%opencl.image2d_wo_t = type opaque

; CHECK-LLVM-LABEL: @nosamp
; CHECK-LLVM: call spir_func <4 x half> @_Z11read_imageh14ocl_image2d_roDv2_i

; CHECK-LLVM-LABEL: @withsamp
; CHECK-LLVM: call spir_func <4 x half> @_Z11read_imageh14ocl_image2d_ro11ocl_samplerDv2_i

; CHECK-LLVM-LABEL: @writehalf
; CHECK-LLVM: call spir_func void @_Z12write_imageh14ocl_image2d_woDv2_iDv4_Dh

; Function Attrs: convergent nounwind
define spir_kernel void @nosamp(%opencl.image2d_ro_t addrspace(1)* %im, <2 x i32> %coord, <4 x half> addrspace(1)* nocapture %res) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func <4 x half> @_Z11read_imageh14ocl_image2d_roDv2_i(%opencl.image2d_ro_t addrspace(1)* %im, <2 x i32> %coord) #3
  store <4 x half> %call, <4 x half> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func <4 x half> @_Z11read_imageh14ocl_image2d_roDv2_i(%opencl.image2d_ro_t addrspace(1)*, <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define spir_kernel void @withsamp(%opencl.image2d_ro_t addrspace(1)* %im, %opencl.sampler_t addrspace(2)* %smp, <2 x i32> %coord, <4 x half> addrspace(1)* nocapture %res) local_unnamed_addr #0 !kernel_arg_addr_space !11 !kernel_arg_access_qual !12 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
entry:
  %call = tail call spir_func <4 x half> @_Z11read_imageh14ocl_image2d_ro11ocl_samplerDv2_i(%opencl.image2d_ro_t addrspace(1)* %im, %opencl.sampler_t addrspace(2)* %smp, <2 x i32> %coord) #3
  store <4 x half> %call, <4 x half> addrspace(1)* %res, align 8, !tbaa !8
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func <4 x half> @_Z11read_imageh14ocl_image2d_ro11ocl_samplerDv2_i(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, <2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define spir_kernel void @writehalf(%opencl.image2d_wo_t addrspace(1)* %im, <2 x i32> %coord, <4 x half> addrspace(1)* nocapture readonly %val) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !16 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = load <4 x half>, <4 x half> addrspace(1)* %val, align 8, !tbaa !8
  tail call spir_func void @_Z12write_imageh14ocl_image2d_woDv2_iDv4_Dh(%opencl.image2d_wo_t addrspace(1)* %im, <2 x i32> %coord, <4 x half> %0) #4
  ret void
}

; Function Attrs: convergent
declare spir_func void @_Z12write_imageh14ocl_image2d_woDv2_iDv4_Dh(%opencl.image2d_wo_t addrspace(1)*, <2 x i32>, <4 x half>) local_unnamed_addr #2

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="64" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind readonly }
attributes #4 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 8.0.0"}
!3 = !{i32 1, i32 0, i32 1}
!4 = !{!"read_only", !"none", !"none"}
!5 = !{!"image2d_t", !"int2", !"half4*"}
!6 = !{!"image2d_t", !"int __attribute__((ext_vector_type(2)))", !"half __attribute__((ext_vector_type(4)))*"}
!7 = !{!"", !"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i32 1, i32 0, i32 0, i32 1}
!12 = !{!"read_only", !"none", !"none", !"none"}
!13 = !{!"image2d_t", !"sampler_t", !"int2", !"half4*"}
!14 = !{!"image2d_t", !"sampler_t", !"int __attribute__((ext_vector_type(2)))", !"half __attribute__((ext_vector_type(4)))*"}
!15 = !{!"", !"", !"", !""}
!16 = !{!"write_only", !"none", !"none"}
