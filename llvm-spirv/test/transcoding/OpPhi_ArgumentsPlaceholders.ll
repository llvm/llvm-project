;;struct Node;
;;typedef struct {
;;    __global struct Node* pNext;
;;} Node;
;;
;;__kernel void verify_linked_lists(__global Node* pNodes)
;;{
;;    __global Node *pNode = pNodes;
;;
;;    for(int j=0; j < 10; j++) {
;;        pNode = pNode->pNext;
;;    }
;;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%struct.Node = type { %struct.Node.0 addrspace(1)* }
%struct.Node.0 = type opaque

; Function Attrs: nounwind
define spir_kernel void @verify_linked_lists(%struct.Node addrspace(1)* %pNodes) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %pNode.0 = phi %struct.Node addrspace(1)* [ %pNodes, %entry ], [ %1, %for.inc ]
  %j.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
;CHECK-SPIRV: Phi {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[BitcastResultId:[0-9]+]] {{[0-9]+}}
;CHECK-SPIRV-NEXT: Phi
;CHECK-LLVM: phi %struct.Node addrspace(1)* [ %pNodes, %entry ], [ [[BitcastResult:%[0-9]+]], %for.inc ]
;CHECK-LLVM-NEXT: phi

  %cmp = icmp slt i32 %j.0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %pNext = getelementptr inbounds %struct.Node, %struct.Node addrspace(1)* %pNode.0, i32 0, i32 0

  %0 = load %struct.Node.0 addrspace(1)*, %struct.Node.0 addrspace(1)* addrspace(1)* %pNext, align 4
  %1 = bitcast %struct.Node.0 addrspace(1)* %0 to %struct.Node addrspace(1)*
;CHECK-SPIRV: Load {{[0-9]+}} [[LoadResultId:[0-9]+]]
;CHECK-SPIRV: Bitcast {{[0-9]+}} [[BitcastResultId]] [[LoadResultId]]
;CHECK-LLVM: [[LoadResult:%[0-9]+]] = load %struct.Node.0 addrspace(1)*, %struct.Node.0 addrspace(1)* addrspace(1)* {{.*}}
;CHECK-LLVM: [[BitcastResult]] = bitcast %struct.Node.0 addrspace(1)* [[LoadResult]] to %struct.Node addrspace(1)*

  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %j.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"Node*"}
!4 = !{!"struct __Node*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}

