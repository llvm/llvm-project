; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t 
; RUN: FileCheck < %t %s
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%struct.Node = type { %struct.Node.0 addrspace(1)* }
; CHECK: 6 TypeOpaque {{[0-9]*}} "struct.Node.0"
%struct.Node.0 = type opaque

; Function Attrs: nounwind readnone
define spir_kernel void @create_linked_lists(%struct.Node addrspace(1)* nocapture %pNodes, i32 addrspace(1)* nocapture %allocation_index, i32 %list_length) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
entry:
  ret void
}

attributes #0 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1, i32 0}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"struct Node*", !"int*", !"int"}
!4 = !{!"", !"volatile ", !""}
!5 = !{!"struct Node*", !"int*", !"int"}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{!""}
