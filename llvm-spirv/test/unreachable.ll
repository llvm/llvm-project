; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 1 Unreachable
; CHECK-LLVM: unreachable

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @unreachable_simple(i32 addrspace(1)* nocapture %in, i32 addrspace(1)* %out) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
  %1 = call spir_func i64 @_Z13get_global_idj(i32 0) #1
  %2 = shl i64 %1, 32
  %3 = ashr exact i64 %2, 32
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %3
  %5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %3
  br label %7
                                                  ; No predecessors!
  unreachable

; <label>:7                                       ; preds = %0
  %8 = load i32, i32 addrspace(1)* %4
  store i32 %8, i32 addrspace(1)* %5
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!6}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!spirv.Generator = !{!9}

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"int*", !"int*"}
!4 = !{!"", !""}
!5 = !{!"int*", !"int*"}
!6 = !{i32 3, i32 102000}
!7 = !{i32 1, i32 2}
!8 = !{}
!9 = !{i16 7, i16 0}
