; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; XFAIL: *

; CHECK-NOT: Name %[[#]] "llvm.masked.gather.v4i32.v4p4"
; CHECK-NOT: Name %[[#]] "llvm.masked.scatter.v4i32.v4p4"

; CHECK-DAG: OpCapability MaskedGatherScatterINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_masked_gather_scatter"

; CHECK-DAG: %[[#TYPEINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYPEPTRINT:]] = OpTypePointer Generic %[[#TYPEINT]]
; CHECK-DAG: %[[#TYPEVECPTR:]] = OpTypeVector %[[#TYPEPTRINT]] 4
; CHECK-DAG: %[[#TYPEVECINT:]] = OpTypeVector %[[#TYPEINT]] 4

; CHECK-DAG: %[[#CONST4:]] = OpConstant %[[#TYPEINT]]  4
; CHECK-DAG: %[[#CONST0:]] = OpConstant %[[#TYPEINT]]  0
; CHECK-DAG: %[[#CONST1:]] = OpConstant %[[#TYPEINT]]  1
; CHECK-DAG: %[[#TRUE:]] = OpConstantTrue %[[#]] 
; CHECK-DAG: %[[#FALSE:]] = OpConstantFalse %[[#]] 
; CHECK-DAG: %[[#MASK1:]] = OpConstantComposite %[[#]] %[[#TRUE]] %[[#FALSE]] %[[#TRUE]] %[[#TRUE]]
; CHECK-DAG: %[[#FILL:]] = OpConstantComposite %[[#]] %[[#CONST4]] %[[#CONST0]] %[[#CONST1]] %[[#CONST0]]
; CHECK-DAG: %[[#MASK2:]] = OpConstantComposite %[[#]] %[[#TRUE]] %[[#TRUE]] %[[#TRUE]] %[[#TRUE]]

; CHECK: %[[#VECGATHER:]] = OpLoad %[[#TYPEVECPTR]] 
; CHECK: %[[#VECSCATTER:]] = OpLoad %[[#TYPEVECPTR]] 
; CHECK: %[[#GATHER:]] = OpMaskedGatherINTEL %[[#TYPEVECINT]] %[[#VECGATHER]] 4 %[[#MASK1]] %[[#FILL]]
; CHECK: OpMaskedScatterINTEL %[[#GATHER]] %[[#VECSCATTER]] 4 %[[#MASK2]]

define spir_kernel void @foo() {
entry:
  %arg0 = alloca <4 x ptr addrspace(4)>
  %arg1 = alloca <4 x ptr addrspace(4)>
  %0 = load <4 x ptr addrspace(4)>, ptr %arg0
  %1 = load <4 x ptr addrspace(4)>, ptr %arg1
  %res = call <4 x i32> @llvm.masked.gather.v4i32.v4p4(<4 x ptr addrspace(4)> %0, i32 4, <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> <i32 4, i32 0, i32 1, i32 0>)
  call void @llvm.masked.scatter.v4i32.v4p4(<4 x i32> %res, <4 x ptr addrspace(4)> %1, i32 4, <4 x i1> splat (i1 true))
  ret void
}

declare <4 x i32> @llvm.masked.gather.v4i32.v4p4(<4 x ptr addrspace(4)>, i32, <4 x i1>, <4 x i32>)
declare void @llvm.masked.scatter.v4i32.v4p4(<4 x i32>, <4 x ptr addrspace(4)>, i32, <4 x i1>)
