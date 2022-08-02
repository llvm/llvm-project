;; struct Node;
;; typedef struct {
;;     __global struct Node* pNext;
;; } Node;
;;
;; __kernel void verify_linked_lists(__global Node* pNodes)
;; {
;;     __global Node *pNode = pNodes;
;;
;;     for(int j=0; j < 10; j++) {
;;         pNode = pNode->pNext;
;;     }
;; }

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

%struct.Node = type { %struct.Node.0 addrspace(1)* }
%struct.Node.0 = type opaque

define spir_kernel void @verify_linked_lists(%struct.Node addrspace(1)* %pNodes) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %pNode.0 = phi %struct.Node addrspace(1)* [ %pNodes, %entry ], [ %1, %for.inc ]
  %j.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
;CHECK-SPIRV: %[[#]] = OpPhi %[[#]] %[[#]] %[[#]] %[[#BitcastResultId:]] %[[#]]
;CHECK-SPIRV-NEXT: OpPhi

  %cmp = icmp slt i32 %j.0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %pNext = getelementptr inbounds %struct.Node, %struct.Node addrspace(1)* %pNode.0, i32 0, i32 0

  %0 = load %struct.Node.0 addrspace(1)*, %struct.Node.0 addrspace(1)* addrspace(1)* %pNext, align 4
  %1 = bitcast %struct.Node.0 addrspace(1)* %0 to %struct.Node addrspace(1)*
;CHECK-SPIRV: %[[#LoadResultId:]] = OpLoad %[[#]]
;CHECK-SPIRV: %[[#BitcastResultId]] = OpBitcast %[[#]] %[[#LoadResultId]]

  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %j.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
