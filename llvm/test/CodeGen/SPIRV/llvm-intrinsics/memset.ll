; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpDecorate %[[#Memset_p0i32:]] LinkageAttributes "spirv.llvm_memset_p0_i32" Export
; CHECK-DAG: OpDecorate %[[#Memset_p3i32:]] LinkageAttributes "spirv.llvm_memset_p3_i32" Export
; CHECK-DAG: OpDecorate %[[#Memset_p1i64:]] LinkageAttributes "spirv.llvm_memset_p1_i64" Export
; CHECK-DAG: OpDecorate %[[#Memset_p1i64:]] LinkageAttributes "spirv.llvm_memset_p1_i64.volatile" Export

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8]]

; CHECK-DAG: %[[#Const4:]] = OpConstant %[[#Int32]] 4
; CHECK: %[[#Int8x4:]] = OpTypeArray %[[#Int8]] %[[#Const4]]

; CHECK-DAG: %[[#Const12:]] = OpConstant %[[#Int32]] 12
; CHECK: %[[#Int8x12:]] = OpTypeArray %[[#Int8]] %[[#Const12]]

; CHECK-DAG: %[[#Const21:]] = OpConstant %[[#Int8]] 21
; CHECK-DAG: %[[#False:]] = OpConstantFalse %[[#]]
; CHECK-DAG: %[[#ConstComp:]] = OpConstantComposite %[[#Int8x4]] %[[#Const21]] %[[#Const21]] %[[#Const21]] %[[#Const21]]
; CHECK-DAG: %[[#ConstNull:]] = OpConstantNull %[[#Int8x12]]
; CHECK: %[[#VarComp:]] = OpVariable %[[#]] UniformConstant %[[#ConstComp]]
; CHECK: %[[#VarNull:]] = OpVariable %[[#]] UniformConstant %[[#ConstNull]]

; CHECK-DAG: %[[#Int8PtrConst:]] = OpTypePointer UniformConstant %[[#Int8]]
; CHECK: %[[#Target:]] = OpBitcast %[[#Int8Ptr]] %[[#]]
; CHECK: %[[#Source:]] = OpBitcast %[[#Int8PtrConst]] %[[#VarNull]]
; CHECK: OpCopyMemorySized %[[#Target]] %[[#Source]] %[[#Const12]] Aligned 4

; CHECK: %[[#SourceComp:]] = OpBitcast %[[#Int8PtrConst]] %[[#VarComp]]
; CHECK: OpCopyMemorySized %[[#]] %[[#SourceComp]] %[[#Const4]] Aligned 4

; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#]] %[[#Memset_p0i32]] %[[#]] %[[#]] %[[#]] %[[#False]]

; CHECK: %[[#Memset_p0i32]] = OpFunction %[[#]]
; CHECK: %[[#Dest:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#Value:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#Len:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#Volatile:]] = OpFunctionParameter %[[#]]

; CHECK: %[[#Entry:]] = OpLabel
; CHECK: %[[#IsZeroLen:]] = OpIEqual %[[#]] %[[#Zero:]] %[[#Len]]
; CHECK: OpBranchConditional %[[#IsZeroLen]] %[[#End:]] %[[#WhileBody:]]

; CHECK: %[[#WhileBody]] = OpLabel
; CHECK: %[[#Offset:]] = OpPhi %[[#]] %[[#Zero]] %[[#Entry]] %[[#OffsetInc:]] %[[#WhileBody]]
; CHECK: %[[#Ptr:]] = OpInBoundsPtrAccessChain %[[#]] %[[#Dest]] %[[#Offset]]
; CHECK: OpStore %[[#Ptr]] %[[#Value]] Aligned 1
; CHECK: %[[#OffsetInc]] = OpIAdd %[[#]] %[[#Offset]] %[[#One:]]
; CHECK: %[[#NotEnd:]] = OpULessThan %[[#]] %[[#OffsetInc]] %[[#Len]]
; CHECK: OpBranchConditional %[[#NotEnd]] %[[#WhileBody]] %[[#End]]

; CHECK: %[[#End]] = OpLabel
; CHECK: OpReturn

; CHECK: OpFunctionEnd

%struct.S1 = type { i32, i32, i32 }

define spir_func void @_Z5foo11v(%struct.S1 addrspace(4)* noalias nocapture sret(%struct.S1 addrspace(4)*) %agg.result, i32 %s1, i64 %s2, i8 %v) {
  %x = alloca [4 x i8]
  %x.bc = bitcast [4 x i8]* %x to i8*
  %a = bitcast %struct.S1 addrspace(4)* %agg.result to i8 addrspace(4)*
  tail call void @llvm.memset.p4i8.i32(i8 addrspace(4)* align 4 %a, i8 0, i32 12, i1 false)
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %x.bc, i8 21, i32 4, i1 false)

  ;; non-const value
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %x.bc, i8 %v, i32 3, i1 false)

  ;; non-const value and size
  tail call void @llvm.memset.p0i8.i32(i8*  align 4 %x.bc, i8 %v, i32 %s1, i1 false)

  ;; Address spaces, non-const value and size
  %b = addrspacecast i8 addrspace(4)* %a to i8 addrspace(3)*
  tail call void @llvm.memset.p3i8.i32(i8 addrspace(3)* align 4 %b, i8 %v, i32 %s1, i1 false)
  %c = addrspacecast i8 addrspace(4)* %a to i8 addrspace(1)*
  tail call void @llvm.memset.p1i8.i64(i8 addrspace(1)* align 4 %c, i8 %v, i64 %s2, i1 false)

  ;; Volatile
  tail call void @llvm.memset.p1i8.i64(i8 addrspace(1)* align 4 %c, i8 %v, i64 %s2, i1 true)
  ret void
}

declare void @llvm.memset.p4i8.i32(i8 addrspace(4)* nocapture, i8, i32, i1)

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)

declare void @llvm.memset.p3i8.i32(i8 addrspace(3)*, i8, i32, i1)

declare void @llvm.memset.p1i8.i64(i8 addrspace(1)*, i8, i64, i1)
