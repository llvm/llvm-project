; The goal of the test is to check that newly inserted `ptrcast` internal
; intrinsic functions for PHI's operands are inserted at the correct
; positions, and don't break rules of instruction domination and PHI nodes
; grouping at top of basic block.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Case1:]] "case1"
; CHECK-DAG: OpName %[[#Case2:]] "case2"
; CHECK-DAG: OpName %[[#Case3:]] "case3"
; CHECK: %[[#Case1]] = OpFunction
; CHECK: OpBranchConditional
; CHECK: OpPhi
; CHECK: OpBranch
; CHECK-COUNT-2: OpBranchConditional
; CHECK: OpFunctionEnd
; CHECK: %[[#Case2]] = OpFunction
; CHECK: OpBranchConditional
; CHECK: OpPhi
; CHECK: OpBranch
; CHECK-COUNT-2: OpBranchConditional
; CHECK: OpFunctionEnd
; CHECK: %[[#Case3]] = OpFunction
; CHECK: OpBranchConditional
; CHECK: OpPhi
; CHECK: OpBranch
; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpBranchConditional
; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpBranchConditional
; CHECK: OpFunctionEnd

%struct1 = type { i64 }
%struct2 = type { i64, i64 }

@.str.1 = private unnamed_addr addrspace(1) constant [3 x i8] c"OK\00", align 1
@.str.2 = private unnamed_addr addrspace(1) constant [6 x i8] c"WRONG\00", align 1

define spir_func void @case1(i1 %b1, i1 %b2, i1 %b3) {
entry:
  br i1 %b1, label %l1, label %l2

l1:
  %str = phi ptr addrspace(1) [ @.str.1, %entry ], [ @.str.2, %l2 ], [ @.str.2, %l3 ]
  br label %exit

l2:
  br i1 %b2, label %l1, label %l3

l3:
  br i1 %b3, label %l1, label %exit

exit:
  ret void
}

define spir_func void @case2(i1 %b1, i1 %b2, i1 %b3, ptr addrspace(1) byval(%struct1) %str1, ptr addrspace(1) byval(%struct2) %str2) {
entry:
  br i1 %b1, label %l1, label %l2

l1:
  %str = phi ptr addrspace(1) [ %str1, %entry ], [ %str2, %l2 ], [ %str2, %l3 ]
  br label %exit

l2:
  br i1 %b2, label %l1, label %l3

l3:
  br i1 %b3, label %l1, label %exit

exit:
  ret void
}

define spir_func void @case3(i1 %b1, i1 %b2, i1 %b3, ptr addrspace(1) byval(%struct1) %_arg_str1, ptr addrspace(1) byval(%struct2) %_arg_str2) {
entry:
  br i1 %b1, label %l1, label %l2

l1:
  %str = phi ptr addrspace(1) [ %_arg_str1, %entry ], [ %str2, %l2 ], [ %str3, %l3 ]
  br label %exit

l2:
  %str2 = getelementptr inbounds %struct2, ptr addrspace(1) %_arg_str2, i32 1
  br i1 %b2, label %l1, label %l3

l3:
  %str3 = getelementptr inbounds %struct2, ptr addrspace(1) %_arg_str2, i32 2
  br i1 %b3, label %l1, label %exit

exit:
  ret void
}
