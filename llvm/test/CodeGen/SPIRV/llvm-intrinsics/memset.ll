; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpDecorate %[[#NonConstMemset:]] LinkageAttributes "spirv.llvm_memset_p3i8_i32"
; CHECK-SPIRV: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-SPIRV: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-SPIRV: %[[#Lenmemset21:]] = OpConstant %[[#]] 4
; CHECK-SPIRV: %[[#Int8x4:]] = OpTypeArray %[[#Int8]] %[[#Lenmemset21]]
; CHECK-SPIRV: %[[#Int8PtrConst:]] = OpTypePointer UniformConstant %[[#Int8]]
; CHECK-SPIRV: %[[#Lenmemset0:]] = OpConstant %[[#Int32]] 12
; CHECK-SPIRV: %[[#Int8x12:]] = OpTypeArray %[[#Int8]] %[[#Lenmemset0]]
; CHECK-SPIRV: %[[#Const21:]] = OpConstant %[[#]] 21
; CHECK-SPIRV: %[[#False:]] = OpConstantFalse %[[#]]
; CHECK-SPIRV: %[[#InitComp:]] = OpConstantComposite %[[#Int8x4]] %[[#Const21]] %[[#Const21]] %[[#Const21]] %[[#Const21]]
; CHECK-SPIRV: %[[#Init:]] = OpConstantNull %[[#Int8x12]]
; CHECK-SPIRV: %[[#ValComp:]] = OpVariable %[[#]] UniformConstant %[[#InitComp]]
; CHECK-SPIRV: %[[#Val:]] = OpVariable %[[#]] UniformConstant %[[#Init]]

; CHECK-SPIRV: %[[#Target:]] = OpBitcast %[[#Int8Ptr]] %[[#]]
; CHECK-SPIRV: %[[#Source:]] = OpBitcast %[[#Int8PtrConst]] %[[#Val]]
; CHECK-SPIRV: OpCopyMemorySized %[[#Target]] %[[#Source]] %[[#Lenmemset0]] Aligned 4

; CHECK-SPIRV: %[[#SourceComp:]] = OpBitcast %[[#Int8PtrConst]] %[[#ValComp]]
; CHECK-SPIRV: OpCopyMemorySized %[[#]] %[[#SourceComp]] %[[#Lenmemset21]] Aligned 4

; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#]] %[[#NonConstMemset]] %[[#]] %[[#]] %[[#]] %[[#False]]

; CHECK-SPIRV: %[[#NonConstMemset]] = OpFunction %[[#]]
; CHECK-SPIRV: %[[#Dest:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#Value:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#Len:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#Volatile:]] = OpFunctionParameter %[[#]]

; CHECK-SPIRV: %[[#Entry:]] = OpLabel
; CHECK-SPIRV: %[[#IsZeroLen:]] = OpIEqual %[[#]] %[[#Zero:]] %[[#Len]]
; CHECK-SPIRV: OpBranchConditional %[[#IsZeroLen]] %[[#End:]] %[[#WhileBody:]]

; CHECK-SPIRV: %[[#WhileBody]] = OpLabel
; CHECK-SPIRV: %[[#Offset:]] = OpPhi %[[#]] %[[#Zero]] %[[#Entry]] %[[#OffsetInc:]] %[[#WhileBody]]
; CHECK-SPIRV: %[[#Ptr:]] = OpInBoundsPtrAccessChain %[[#]] %[[#Dest]] %[[#Offset]]
; CHECK-SPIRV: OpStore %[[#Ptr]] %[[#Value]] Aligned 1
; CHECK-SPIRV: %[[#OffsetInc]] = OpIAdd %[[#]] %[[#Offset]] %[[#One:]]
; CHECK-SPIRV: %[[#NotEnd:]] = OpULessThan %[[#]] %[[#OffsetInc]] %[[#Len]]
; CHECK-SPIRV: OpBranchConditional %[[#NotEnd]] %[[#WhileBody]] %[[#End]]

; CHECK-SPIRV: %[[#End]] = OpLabel
; CHECK-SPIRV: OpReturn

; CHECK-SPIRV: OpFunctionEnd

%struct.S1 = type { i32, i32, i32 }

define spir_func void @_Z5foo11v(%struct.S1 addrspace(4)* noalias nocapture sret(%struct.S1 addrspace(4)*) %agg.result, i32 %s1, i64 %s2, i8 %v) {
  %x = alloca [4 x i8]
  %x.bc = bitcast [4 x i8]* %x to i8*
  %1 = bitcast %struct.S1 addrspace(4)* %agg.result to i8 addrspace(4)*
  tail call void @llvm.memset.p4i8.i32(i8 addrspace(4)* align 4 %1, i8 0, i32 12, i1 false)
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %x.bc, i8 21, i32 4, i1 false)

  ;; non-const value
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %x.bc, i8 %v, i32 3, i1 false)

  ;; non-const value and size
  tail call void @llvm.memset.p0i8.i32(i8*  align 4 %x.bc, i8 %v, i32 %s1, i1 false)

  ;; Address spaces, non-const value and size
  %a = addrspacecast i8 addrspace(4)* %1 to i8 addrspace(3)*
  tail call void @llvm.memset.p3i8.i32(i8 addrspace(3)* align 4 %a, i8 %v, i32 %s1, i1 false)
  %b = addrspacecast i8 addrspace(4)* %1 to i8 addrspace(1)*
  tail call void @llvm.memset.p1i8.i64(i8 addrspace(1)* align 4 %b, i8 %v, i64 %s2, i1 false)

  ;; Volatile
  tail call void @llvm.memset.p1i8.i64(i8 addrspace(1)* align 4 %b, i8 %v, i64 %s2, i1 true)
  ret void
}

declare void @llvm.memset.p4i8.i32(i8 addrspace(4)* nocapture, i8, i32, i1)

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)

declare void @llvm.memset.p3i8.i32(i8 addrspace(3)*, i8, i32, i1)

declare void @llvm.memset.p1i8.i64(i8 addrspace(1)*, i8, i64, i1)
