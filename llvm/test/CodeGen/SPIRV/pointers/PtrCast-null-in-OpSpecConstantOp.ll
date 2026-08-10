; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[GenPtr:.*]] = OpTypePointer Generic %[[Char]]
; CHECK-DAG: %[[Array:.*]] = OpTypeArray %[[GenPtr]] %[[#]]
; CHECK-DAG: %[[Struct:.*]] = OpTypeStruct %[[Array]]
; CHECK-DAG: %[[Null:.*]] = OpConstantNull %[[GenPtr]]
; CHECK-DAG: %[[R:.*]] = OpConstantComposite %[[Array]] %[[Null]]
; CHECK-DAG: %[[#]] = OpConstantComposite %[[Struct]] %[[R]]

@G1 = addrspace(1) constant { [1 x ptr addrspace(4)] } { [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4))] }
@G2 = addrspace(1) constant { [1 x ptr addrspace(4)] } { [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) null to ptr addrspace(4))] }

define void @foo() {
entry:
  ret void
}
