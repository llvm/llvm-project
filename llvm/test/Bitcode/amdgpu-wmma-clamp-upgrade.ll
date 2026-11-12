; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define amdgpu_ps void @swmmac_i32_16x16x128_iu8(<8 x i32> %A, <16 x i32> %B, <8 x i32> %C, i16 %Index, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_ps void @swmmac_i32_16x16x128_iu8(
; CHECK-SAME: <8 x i32> [[A:%.*]], <16 x i32> [[B:%.*]], <8 x i32> [[C:%.*]], i16 [[INDEX:%.*]], ptr addrspace(1) [[OUT:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x128.iu8.v8i32.v8i32.v16i32.i16(i1 false, <8 x i32> [[A]], i1 false, <16 x i32> [[B]], <8 x i32> [[C]], i16 [[INDEX]], i1 false, i1 false, i1 false)
; CHECK-NEXT:    store <8 x i32> [[TMP1]], ptr addrspace(1) [[OUT]], align 32
; CHECK-NEXT:    ret void
;
  %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x128.iu8.v8i32.v8i32.v16i32.i16(i1 0, <8 x i32> %A, i1 0, <16 x i32> %B, <8 x i32> %C, i16 %Index, i1 false, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @wmma_i32_16x16x64_iu8(<8 x i32> %A, <8 x i32> %B, <8 x i32> %C, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_kernel void @wmma_i32_16x16x64_iu8(
; CHECK-SAME: <8 x i32> [[A:%.*]], <8 x i32> [[B:%.*]], <8 x i32> [[C:%.*]], ptr addrspace(1) [[OUT:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(i1 false, <8 x i32> [[A]], i1 false, <8 x i32> [[B]], <8 x i32> [[C]], i1 false, i1 false, i1 false)
; CHECK-NEXT:    store <8 x i32> [[TMP1]], ptr addrspace(1) [[OUT]], align 32
; CHECK-NEXT:    ret void
;
  %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(i1 0, <8 x i32> %A, i1 0, <8 x i32> %B, <8 x i32> %C, i1 false, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}
