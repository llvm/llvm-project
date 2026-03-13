; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Array:.*]] = OpTypeArray %[[#]] %[[#]]
; CHECK-DAG: %[[Struct:.*]] = OpTypeStruct %[[Array]]
; CHECK-DAG: %[[Zero:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[Null:.*]] = OpConstantNull %[[Zero]]
; CHECK-DAG: %[[R:.*]] = OpConstantComposite %[[Array]] %[[Null]]
; CHECK-DAG: %[[#]] = OpConstantComposite %[[Struct]] %[[R]]

@G1 = addrspace(1) constant { [1 x ptr addrspace(4)] } { [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4))] }
@G2 = addrspace(1) constant { [1 x ptr addrspace(4)] } { [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) null to ptr addrspace(4))] }

define void @foo() {
entry:
  ret void
}
