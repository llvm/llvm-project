; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%struct.A = type { i32, %struct.C }
%struct.C = type { i32, %struct.B }
%struct.B = type { i32, %struct.A addrspace(4)* }
%struct.Node = type { %struct.Node addrspace(1)*, i32 }

; CHECK-SPIRV: 3 TypeForwardPointer [[AFwdPtr:[0-9]+]] [[ASC:[0-9]+]]
; CHECK-SPIRV: 3 TypeForwardPointer [[NodeFwdPtr:[0-9]+]] [[NodeSC:[0-9]+]]
; CHECK-SPIRV: 4 TypeInt [[IntID:[0-9]+]] 32 0
; CHECK-SPIRV: 4 TypeStruct [[BID:[0-9]+]] {{[0-9]+}} [[AFwdPtr]]
; CHECK-SPIRV: 4 TypeStruct [[CID:[0-9]+]] {{[0-9]+}} [[BID]]
; CHECK-SPIRV: 4 TypeStruct [[AID:[0-9]+]] {{[0-9]+}} [[CID]]
; CHECK-SPIRV: 4 TypePointer [[AFwdPtr]] [[ASC]] [[AID:[0-9]+]]
; CHECK-SPIRV: 4 TypeStruct [[NodeID:[0-9]+]] [[NodeFwdPtr]]
; CHECK-SPIRV: 4 TypePointer [[NodeFwdPtr]] [[NodeSC]] [[NodeID]]

; CHECK-LLVM: %struct.A = type { i32, %struct.C }
; CHECK-LLVM: %struct.C = type { i32, %struct.B }
; CHECK-LLVM: %struct.B = type { i32, %struct.A addrspace(4)* }
; CHECK-LLVM: %struct.Node = type { %struct.Node addrspace(1)*, i32 }

; Function Attrs: nounwind
define spir_kernel void @test(%struct.A addrspace(1)* %result, %struct.Node addrspace(1)* %node) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="true" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!8}
!opencl.used.extensions = !{!9}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!9}
!llvm.ident = !{!10}

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"struct A*", !"struct Node*"}
!4 = !{!"struct A*", !"struct Node*"}
!5 = !{!"", !""}
!6 = !{!"result", !"node"}
!7 = !{i32 1, i32 2}
!8 = !{i32 2, i32 0}
!9 = !{}
!10 = !{!"clang version 3.6.1 "}
