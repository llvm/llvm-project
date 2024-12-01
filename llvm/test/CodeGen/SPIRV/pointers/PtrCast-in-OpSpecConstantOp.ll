; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[F:.*]] "F"
; CHECK-DAG: OpName %[[B:.*]] "B"
; CHECK-DAG: OpName %[[G1:.*]] "G1"
; CHECK-DAG: OpName %[[G2:.*]] "G2"
; CHECK-DAG: OpName %[[X:.*]] "X"
; CHECK-DAG: OpName %[[Y:.*]] "Y"
; CHECK-DAG: OpName %[[G3:.*]] "G3"
; CHECK-DAG: OpName %[[G4:.*]] "G4"

; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[GenPtrChar:.*]] = OpTypePointer Generic %[[Char]]
; CHECK-DAG: %[[CWPtrChar:.*]] = OpTypePointer CrossWorkgroup %[[Char]]
; CHECK-DAG: %[[Arr1:.*]] = OpTypeArray %[[CWPtrChar]] %[[#]]
; CHECK-DAG: %[[Struct1:.*]] = OpTypeStruct %8
; CHECK-DAG: %[[Arr2:.*]] = OpTypeArray %[[GenPtrChar]] %[[#]]
; CHECK-DAG: %[[Struct2:.*]] = OpTypeStruct %[[Arr2]]
; CHECK-DAG: %[[GenPtr:.*]] = OpTypePointer Generic %[[Int]]
; CHECK-DAG: %[[CWPtr:.*]] = OpTypePointer CrossWorkgroup %[[Int]]
; CHECK-DAG: %[[WPtr:.*]] = OpTypePointer Workgroup %[[Int]]

; CHECK-DAG: %[[F]] = OpVariable %[[CWPtr]] CrossWorkgroup %[[#]]
; CHECK-DAG: %[[GenF:.*]] = OpSpecConstantOp %[[GenPtrChar]] 121 %[[F]]
; CHECK-DAG: %[[B]] = OpVariable %[[CWPtr]] CrossWorkgroup %[[#]]
; CHECK-DAG: %[[GenB:.*]] = OpSpecConstantOp %[[GenPtrChar]] 121 %[[B]]
; CHECK-DAG: %[[GenFB:.*]] = OpConstantComposite %[[Arr2]] %[[GenF]] %[[GenB]]
; CHECK-DAG: %[[GenBF:.*]] = OpConstantComposite %[[Arr2]] %[[GenB]] %[[GenF]]
; CHECK-DAG: %[[CG1:.*]] = OpConstantComposite %[[Struct2]] %[[GenFB]]
; CHECK-DAG: %[[CG2:.*]] = OpConstantComposite %[[Struct2]] %[[GenBF]]

; CHECK-DAG: %[[X]] = OpVariable %[[WPtr]] Workgroup %[[#]]
; CHECK-DAG: %[[GenX:.*]] = OpSpecConstantOp %[[GenPtr]] 121 %[[X]]
; CHECK-DAG: %[[CWX:.*]] = OpSpecConstantOp %[[CWPtrChar]] 122 %[[GenX]]
; CHECK-DAG: %[[Y]] = OpVariable %[[WPtr]] Workgroup %[[#]]
; CHECK-DAG: %[[GenY:.*]] = OpSpecConstantOp %[[GenPtr]] 121 %[[Y]]
; CHECK-DAG: %[[CWY:.*]] = OpSpecConstantOp %[[CWPtrChar]] 122 %[[GenY]]
; CHECK-DAG: %[[CWXY:.*]] = OpConstantComposite %[[Arr1]] %[[CWX]] %[[CWY]]
; CHECK-DAG: %[[CWYX:.*]] = OpConstantComposite %[[Arr1]] %[[CWY]] %[[CWX]]
; CHECK-DAG: %[[CG3:.*]] = OpConstantComposite %[[Struct1]] %[[CWXY]]
; CHECK-DAG: %[[CG4:.*]] = OpConstantComposite %[[Struct1]] %[[CWYX]]

; CHECK-DAG: %[[G4]] = OpVariable %[[#]] CrossWorkgroup %[[CG4]]
; CHECK-DAG: %[[G3]] = OpVariable %[[#]] CrossWorkgroup %[[CG3]]
; CHECK-DAG: %[[G2]] = OpVariable %[[#]] CrossWorkgroup %[[CG2]]
; CHECK-DAG: %[[G1]] = OpVariable %[[#]] CrossWorkgroup %[[CG1]]

@F = addrspace(1) constant i32 0
@B = addrspace(1) constant i32 1
@G1 = addrspace(1) constant { [2 x ptr addrspace(4)] } { [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @F to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @B to ptr addrspace(4))] }
@G2 = addrspace(1) constant { [2 x ptr addrspace(4)] } { [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @B to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @F to ptr addrspace(4))] }

@X = addrspace(3) constant i32 0
@Y = addrspace(3) constant i32 1
@G3 = addrspace(1) constant { [2 x ptr addrspace(1)] } { [2 x ptr addrspace(1)] [ptr addrspace(1) addrspacecast (ptr addrspace(3) @X to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr addrspace(3) @Y to ptr addrspace(1))] }
@G4 = addrspace(1) constant { [2 x ptr addrspace(1)] } { [2 x ptr addrspace(1)] [ptr addrspace(1) addrspacecast (ptr addrspace(3) @Y to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr addrspace(3) @X to ptr addrspace(1))] }

define void @foo() {
entry:
  ret void
}
