; RUN: llc -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#GV:]] "v"
; CHECK-DAG: OpName %[[#FunBar:]] "bar"
; CHECK-DAG: OpDecorate %[[#GV]] LinkageAttributes "v" Export
; CHECK-DAG: OpDecorate %[[#GV]] Constant
; CHECK-DAG: OpDecorateId %[[#GV]] AlignmentId %[[#C8:]]
; CHECK-DAG: OpDecorateId %[[#GV]] MaxByteOffsetId %[[#C0:]]
; CHECK-DAG: %[[#C8]] = OpConstant %[[#]] 8
; CHECK-DAG: %[[#C0]] = OpConstantNull %[[#]]
; CHECK-DAG: OpDecorate %[[#Idx:]] UserSemantic "SemanticValue"
; CHECK: %[[#FunBar]] = OpFunction
; CHECK: %[[#Idx]] = OpInBoundsPtrAccessChain

@v = addrspace(1) global i32 0, !spirv.Decorations !0

define spir_kernel void @foo() {
entry:
  %pv = load ptr addrspace(1), ptr addrspace(1) @v
  store i32 3, ptr addrspace(1) %pv
  ret void
}

define spir_kernel void @bar(ptr addrspace(1) %arg) {
entry:
  %idx = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1, !spirv.Decorations !3
  ret void
}

!0 = !{!1, !2, !5, !6}
!1 = !{i32 22}                     ; 22 is Constant decoration
!2 = !{i32 41, !"v", i32 0}        ; 41 is LinkageAttributes decoration with 2 extra operands
!3 = !{!4}
!4 = !{i32 5635, !"SemanticValue"} ; 5635 is UserSemantic decoration
!5 = !{i32 46, i32 8}              ; 46 is AlignmentId decoration
!6 = !{i32 47, i32 0}              ; 47 is MaxByteOffsetId decoration
