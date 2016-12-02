; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; This is used to crash in LiveIntervalAnalysis via SILoadStoreOptimizer
; while fixing up the merge of two ds_write instructions.

@tess_lds = external addrspace(3) global [8192 x i32]

; CHECK-LABEL: {{^}}main:
; CHECK: ds_write2_b32
; CHECK: v_mov_b32_e32 v1, v0
; CHECK: tbuffer_store_format_xyzw v[0:3],
define amdgpu_vs void @main(i32 inreg %arg) {
main_body:
  %tmp = load float, float addrspace(3)* undef, align 4
  %tmp1 = load float, float addrspace(3)* undef, align 4
  store float %tmp, float addrspace(3)* null, align 4
  %tmp2 = bitcast float %tmp to i32
  %tmp3 = add nuw nsw i32 0, 1
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = getelementptr [8192 x i32], [8192 x i32] addrspace(3)* @tess_lds, i64 0, i64 %tmp4
  %tmp6 = bitcast i32 addrspace(3)* %tmp5 to float addrspace(3)*
  store float %tmp1, float addrspace(3)* %tmp6, align 4
  %tmp7 = bitcast float %tmp1 to i32
  %tmp8 = insertelement <4 x i32> undef, i32 %tmp2, i32 0
  %tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp7, i32 1
  %tmp10 = insertelement <4 x i32> %tmp9, i32 undef, i32 2
  %tmp11 = insertelement <4 x i32> %tmp10, i32 undef, i32 3
  call void @llvm.SI.tbuffer.store.v4i32(<16 x i8> undef, <4 x i32> %tmp11, i32 4, i32 undef, i32 %arg, i32 0, i32 14, i32 4, i32 1, i32 0, i32 1, i32 1, i32 0)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.SI.tbuffer.store.v4i32(<16 x i8>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind }
