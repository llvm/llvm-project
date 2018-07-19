; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s
; We may have subregister live ranges that are undefined on some paths. The
; verifier should not complain about this.


; CHECK-LABEL: {{^}}func:
define amdgpu_kernel void @func() #0 {
B0:
  br i1 undef, label %B1, label %B2

B1:
  br label %B2

B2:
  %v0 = phi <4 x float> [ zeroinitializer, %B1 ], [ <float 0.0, float 0.0, float 0.0, float undef>, %B0 ]
  br i1 undef, label %B30.1, label %B30.2

B30.1:
  %sub = fsub <4 x float> %v0, undef
  br label %B30.2

B30.2:
  %v3 = phi <4 x float> [ %sub, %B30.1 ], [ %v0, %B2 ]
  %ve0 = extractelement <4 x float> %v3, i32 0
  store float %ve0, float addrspace(3)* undef, align 4
  ret void
}

; FIXME: Extra undef subregister copy should be removed before
; overwritten with defined copy
; CHECK-LABEL: {{^}}valley_partially_undef_copy:
define amdgpu_ps float @valley_partially_undef_copy() #0 {
bb:
  %tmp = load volatile i32, i32 addrspace(1)* undef, align 4
  %tmp1 = load volatile i32, i32 addrspace(1)* undef, align 4
  %tmp2 = insertelement <4 x i32> undef, i32 %tmp1, i32 0
  %tmp3 = insertelement <4 x i32> %tmp2, i32 %tmp1, i32 1
  %tmp3.cast = bitcast <4 x i32> %tmp3 to <4 x float>
  %tmp4 = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float> %tmp3.cast, <8 x i32> undef, <4 x i32> undef, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp5 = extractelement <4 x float> %tmp4, i32 0
  %tmp6 = fmul float %tmp5, undef
  %tmp7 = fadd float %tmp6, %tmp6
  %tmp8 = insertelement <4 x i32> %tmp2, i32 %tmp, i32 1
  store <4 x i32> %tmp8, <4 x i32> addrspace(1)* undef, align 16
  store float %tmp7, float addrspace(1)* undef, align 4
  br label %bb9

bb9:                                              ; preds = %bb9, %bb
  %tmp10 = icmp eq i32 %tmp, 0
  br i1 %tmp10, label %bb9, label %bb11

bb11:                                             ; preds = %bb9
  store <4 x i32> %tmp2, <4 x i32> addrspace(1)* undef, align 16
  ret float undef
}

; FIXME: Should be able to remove the undef copies

; CHECK-LABEL: {{^}}partially_undef_copy:
; CHECK: v_mov_b32_e32 v5, 5
; CHECK: v_mov_b32_e32 v6, 6

; CHECK: v_mov_b32_e32 v[[OUTPUT_LO:[0-9]+]], v5

; Undef copy
; CHECK: v_mov_b32_e32 v1, v6

; undef copy
; CHECK: v_mov_b32_e32 v2, v7

; CHECK: v_mov_b32_e32 v[[OUTPUT_HI:[0-9]+]], v8
; CHECK: v_mov_b32_e32 v[[OUTPUT_LO]], v6

; CHECK: buffer_store_dwordx4 v{{\[}}[[OUTPUT_LO]]:[[OUTPUT_HI]]{{\]}}
define amdgpu_kernel void @partially_undef_copy() #0 {
  %tmp0 = call i32 asm sideeffect "v_mov_b32_e32 v5, 5", "={v5}"()
  %tmp1 = call i32 asm sideeffect "v_mov_b32_e32 v6, 6", "={v6}"()

  %partially.undef.0 = insertelement <4 x i32> undef, i32 %tmp0, i32 0
  %partially.undef.1 = insertelement <4 x i32> %partially.undef.0, i32 %tmp1, i32 0

  store volatile <4 x i32> %partially.undef.1, <4 x i32> addrspace(1)* undef, align 16
  tail call void asm sideeffect "v_nop", "v={v[5:8]}"(<4 x i32> %partially.undef.0)
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
