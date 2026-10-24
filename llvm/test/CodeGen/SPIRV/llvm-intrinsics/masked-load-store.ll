; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; FIXME: enabled on Vulkan env, when legalization of vectors > 4 elements is
; fully supported.

; Verify that llvm.masked.load and llvm.masked.store lower correctly.
; SPIR-V reports branch divergence, so ScalarizeMaskedMemIntrin extracts each
; mask lane directly with OpCompositeExtract instead of reconstructing an
; integer bitmask, since scalar bit tests offer no benefit under per-lane
; divergent execution.

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#BVEC8:]] = OpTypeVector %[[#BOOL]] 8
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#VOID:]] = OpTypeVoid

; The scalarization of llvm.masked.load/store extracts each predicate
; directly from the boolean mask vector.
;
; CHECK:   OpFunction
; CHECK:   %[[#ML_MASK:]] = OpFunctionParameter %[[#BVEC8]]
; CHECK:   %[[#ML_P0:]] = OpCompositeExtract %[[#BOOL]] %[[#ML_MASK]] 0
; CHECK:   OpBranchConditional %[[#ML_P0]]
; CHECK:   %[[#ML_LD:]] = OpLoad %[[#I8]]
; CHECK:   %[[#ML_INS:]] = OpCompositeInsert %{{.*}} %[[#ML_LD]] %{{.*}} 0
; CHECK:   OpPhi %{{.*}} %[[#ML_INS]]
; CHECK:   %[[#ML_P1:]] = OpCompositeExtract %[[#BOOL]] %[[#ML_MASK]] 1
; CHECK:   OpBranchConditional %[[#ML_P1]]
; CHECK:   OpStore %{{.*}} %{{.*}}
define void @masked_load_v8i8(<8 x i1> %mask) {
  %v = call <8 x i8> @llvm.masked.load.v8i8.p1(ptr addrspace(1) null, <8 x i1> %mask, <8 x i8> zeroinitializer)
  store <8 x i8> %v, ptr addrspace(3) null, align 1
  ret void
}

; CHECK:   OpFunction
; CHECK:   %[[#MS_VAL:]] = OpFunctionParameter
; CHECK:   %[[#MS_MASK:]] = OpFunctionParameter %[[#BVEC8]]
; CHECK:   %[[#MS_P0:]] = OpCompositeExtract %[[#BOOL]] %[[#MS_MASK]] 0
; CHECK:   OpBranchConditional %[[#MS_P0]]
; CHECK:   %[[#MS_ELEM:]] = OpCompositeExtract %[[#I8]] %[[#MS_VAL]] 0
; CHECK:   OpStore %{{.*}} %[[#MS_ELEM]]
define void @masked_store_v8i8(<8 x i8> %val, <8 x i1> %mask) {
  call void @llvm.masked.store.v8i8.p1(<8 x i8> %val, ptr addrspace(1) null, <8 x i1> %mask)
  ret void
}

declare <8 x i8> @llvm.masked.load.v8i8.p1(ptr addrspace(1), <8 x i1>, <8 x i8>)
declare void @llvm.masked.store.v8i8.p1(<8 x i8>, ptr addrspace(1), <8 x i1>)
