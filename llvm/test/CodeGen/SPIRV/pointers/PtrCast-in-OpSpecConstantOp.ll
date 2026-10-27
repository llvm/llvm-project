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
; CHECK-DAG: %[[Struct1:.*]] = OpTypeStruct %[[Arr1]]
; CHECK-DAG: %[[Arr2:.*]] = OpTypeArray %[[GenPtrChar]] %[[#]]
; CHECK-DAG: %[[Struct2:.*]] = OpTypeStruct %[[Arr2]]
; CHECK-DAG: %[[CWPtr:.*]] = OpTypePointer CrossWorkgroup %[[Int]]
; CHECK-DAG: %[[WPtr:.*]] = OpTypePointer Workgroup %[[Int]]
; CHECK-DAG: %[[WPtrChar:.*]] = OpTypePointer Workgroup %[[Char]]

; CHECK-DAG: %[[F]] = OpVariable %[[CWPtr]] CrossWorkgroup %[[#]]
; CHECK-DAG: %[[CharF:.*]] = OpSpecConstantOp %[[CWPtrChar]] Bitcast %[[F]]
; CHECK-DAG: %[[GenF:.*]] = OpSpecConstantOp %[[GenPtrChar]] PtrCastToGeneric %[[CharF]]
; CHECK-DAG: %[[B]] = OpVariable %[[CWPtr]] CrossWorkgroup %[[#]]
; CHECK-DAG: %[[CharB:.*]] = OpSpecConstantOp %[[CWPtrChar]] Bitcast %[[B]]
; CHECK-DAG: %[[GenB:.*]] = OpSpecConstantOp %[[GenPtrChar]] PtrCastToGeneric %[[CharB]]
; CHECK-DAG: %[[GenFB:.*]] = OpSpecConstantComposite %[[Arr2]] %[[GenF]] %[[GenB]]
; CHECK-DAG: %[[GenBF:.*]] = OpSpecConstantComposite %[[Arr2]] %[[GenB]] %[[GenF]]
; CHECK-DAG: %[[CG1:.*]] = OpSpecConstantComposite %[[Struct2]] %[[GenFB]]
; CHECK-DAG: %[[CG2:.*]] = OpSpecConstantComposite %[[Struct2]] %[[GenBF]]

; CHECK-DAG: %[[X]] = OpVariable %[[WPtr]] Workgroup %[[#]]
; CHECK-DAG: %[[CharX:.*]] = OpSpecConstantOp %[[WPtrChar]] Bitcast %[[X]]
; CHECK-DAG: %[[GenX:.*]] = OpSpecConstantOp %[[GenPtrChar]] PtrCastToGeneric %[[CharX]]
; CHECK-DAG: %[[CWX:.*]] = OpSpecConstantOp %[[CWPtrChar]] GenericCastToPtr %[[GenX]]
; CHECK-DAG: %[[Y]] = OpVariable %[[WPtr]] Workgroup %[[#]]
; CHECK-DAG: %[[CharY:.*]] = OpSpecConstantOp %[[WPtrChar]] Bitcast %[[Y]]
; CHECK-DAG: %[[GenY:.*]] = OpSpecConstantOp %[[GenPtrChar]] PtrCastToGeneric %[[CharY]]
; CHECK-DAG: %[[CWY:.*]] = OpSpecConstantOp %[[CWPtrChar]] GenericCastToPtr %[[GenY]]
; CHECK-DAG: %[[CWXY:.*]] = OpSpecConstantComposite %[[Arr1]] %[[CWX]] %[[CWY]]
; CHECK-DAG: %[[CWYX:.*]] = OpSpecConstantComposite %[[Arr1]] %[[CWY]] %[[CWX]]
; CHECK-DAG: %[[CG3:.*]] = OpSpecConstantComposite %[[Struct1]] %[[CWXY]]
; CHECK-DAG: %[[CG4:.*]] = OpSpecConstantComposite %[[Struct1]] %[[CWYX]]

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
