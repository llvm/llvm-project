; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; FIXME: enabled on Vulkan env, when legalization of vectors > 4 elements is
; fully supported.

; Verify that llvm.masked.load and llvm.masked.store lower correctly.
; ScalarizeMaskedMemIntrin expands these into scalar conditional loads/stores
; with <N x i1> -> iN bitcasts for the mask, which must be decomposed by the
; backend as SPIR-V don't doesn't allow bitcasts to/from boolean vectors.

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#BVEC8:]] = OpTypeVector %[[#BOOL]] 8
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#VOID:]] = OpTypeVoid

; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#I8]]
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I8]] 1

; The scalarization of llvm.masked.load/store produces <8 x i1> -> i8 bitcast.
; Verify the mask decomposition and scalarized conditional loads compile.
;
; CHECK:   OpFunction
; CHECK:   %[[#ML_MASK:]] = OpFunctionParameter %[[#BVEC8]]
; CHECK:   %[[#ML_E0:]] = OpCompositeExtract %[[#BOOL]] %[[#ML_MASK]] 0
; CHECK:   %[[#ML_S0:]] = OpSelect %[[#I8]] %[[#ML_E0]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#ML_OR0:]] = OpBitwiseOr %[[#I8]] %[[#ZERO]] %[[#ML_S0]]
; CHECK:   %[[#ML_E1:]] = OpCompositeExtract %[[#BOOL]] %[[#ML_MASK]] 1
; CHECK:   %[[#ML_S1:]] = OpSelect %[[#I8]] %[[#ML_E1]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#ML_SHL1:]] = OpShiftLeftLogical %[[#I8]] %[[#ML_S1]] %[[#ONE]]
; CHECK:   %[[#ML_OR1:]] = OpBitwiseOr %[[#I8]] %[[#ML_OR0]] %[[#ML_SHL1]]
; CHECK:   %[[#ML_A0:]] = OpBitwiseAnd %[[#I8]] %{{.*}} %[[#ONE]]
; CHECK:   %[[#ML_C0:]] = OpINotEqual %[[#BOOL]] %[[#ML_A0]] %[[#ZERO]]
; CHECK:   OpBranchConditional %[[#ML_C0]]
; CHECK:   %[[#ML_LD:]] = OpLoad %[[#I8]]
; CHECK:   %[[#ML_INS:]] = OpCompositeInsert %{{.*}} %[[#ML_LD]] %{{.*}} 0
; CHECK:   OpPhi %{{.*}} %[[#ML_INS]]
; CHECK:   OpStore %{{.*}} %{{.*}}
define void @masked_load_v8i8(<8 x i1> %mask) {
  %v = call <8 x i8> @llvm.masked.load.v8i8.p1(ptr addrspace(1) null, <8 x i1> %mask, <8 x i8> zeroinitializer)
  store <8 x i8> %v, ptr addrspace(3) null, align 1
  ret void
}

; CHECK:   OpFunction
; CHECK:   %[[#MS_VAL:]] = OpFunctionParameter
; CHECK:   %[[#MS_MASK:]] = OpFunctionParameter %[[#BVEC8]]
; CHECK:   %[[#MS_E0:]] = OpCompositeExtract %[[#BOOL]] %[[#MS_MASK]] 0
; CHECK:   %[[#MS_S0:]] = OpSelect %[[#I8]] %[[#MS_E0]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#MS_OR0:]] = OpBitwiseOr %[[#I8]] %[[#ZERO]] %[[#MS_S0]]
; CHECK:   %[[#MS_E1:]] = OpCompositeExtract %[[#BOOL]] %[[#MS_MASK]] 1
; CHECK:   %[[#MS_S1:]] = OpSelect %[[#I8]] %[[#MS_E1]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#MS_SHL1:]] = OpShiftLeftLogical %[[#I8]] %[[#MS_S1]] %[[#ONE]]
; CHECK:   %[[#MS_OR1:]] = OpBitwiseOr %[[#I8]] %[[#MS_OR0]] %[[#MS_SHL1]]
; CHECK:   %[[#MS_A0:]] = OpBitwiseAnd %[[#I8]] %{{.*}} %[[#ONE]]
; CHECK:   %[[#MS_C0:]] = OpINotEqual %[[#BOOL]] %[[#MS_A0]] %[[#ZERO]]
; CHECK:   OpBranchConditional %[[#MS_C0]]
; CHECK:   %[[#MS_ELEM:]] = OpCompositeExtract %[[#I8]] %[[#MS_VAL]] 0
; CHECK:   OpStore %{{.*}} %[[#MS_ELEM]]
define void @masked_store_v8i8(<8 x i8> %val, <8 x i1> %mask) {
  call void @llvm.masked.store.v8i8.p1(<8 x i8> %val, ptr addrspace(1) null, <8 x i1> %mask)
  ret void
}

declare <8 x i8> @llvm.masked.load.v8i8.p1(ptr addrspace(1), <8 x i1>, <8 x i8>)
declare void @llvm.masked.store.v8i8.p1(<8 x i8>, ptr addrspace(1), <8 x i1>)
