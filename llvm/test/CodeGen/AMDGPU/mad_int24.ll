; RUN: llc < %s -mtriple=amdgcn -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=FUNC
; RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=FUNC
; RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc < %s -mtriple=r600 -mcpu=cayman | FileCheck %s --check-prefix=CM --check-prefix=FUNC

; FUNC-LABEL: {{^}}i32_mad24:
; Signed 24-bit multiply is not supported on pre-Cayman GPUs.
; EG: MULLO_INT
; CM: MULLO_INT
; GCN: s_bfe_i32
; GCN: s_bfe_i32
; GCN: s_mul_i32
; GCN: s_add_i32
define amdgpu_kernel void @i32_mad24(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = ashr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = ashr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  %3 = add i32 %2, %c
  store i32 %3, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mad24_known_bits_destroyed:
; GCN: s_waitcnt
; GCN-NEXT: v_mad_i32_i24
; GCN-NEXT: v_mul_i32_i24
; GCN-NEXT: s_setpc_b64
define i32 @mad24_known_bits_destroyed(i32 %a, i32 %b, i32 %c) {

  %shl.0 = shl i32 %a, 8
  %sra.0 = ashr i32 %shl.0, 8
  %shl.1 = shl i32 %b, 8
  %sra.1 = ashr i32 %shl.1, 8

  %mul0 = mul nsw i32 %sra.0, %sra.1
  %add0 = add nsw i32 %mul0, %c

  %shl.2 = shl i32 %add0, 8
  %sra.2 = ashr i32 %shl.2, 8

  %shl.3 = shl i32 %sra.0, 8
  %sra.3 = ashr i32 %shl.3, 8

  %mul1 = mul nsw i32 %sra.2, %sra.3
  ret i32 %mul1
}

; GCN-LABEL: {{^}}mad24_intrin_known_bits_destroyed:
; GCN: s_waitcnt
; GCN-NEXT: v_mad_i32_i24
; GCN-NEXT: v_mul_i32_i24
; GCN-NEXT: s_setpc_b64
define i32 @mad24_intrin_known_bits_destroyed(i32 %a, i32 %b, i32 %c) {
  %shl.0 = shl i32 %a, 8
  %sra.0 = ashr i32 %shl.0, 8
  %shl.1 = shl i32 %b, 8
  %sra.1 = ashr i32 %shl.1, 8

  %mul0 = call i32 @llvm.amdgcn.mul.i24(i32 %sra.0, i32 %sra.1)
  %add0 = add nsw i32 %mul0, %c

  %shl.2 = shl i32 %add0, 8
  %sra.2 = ashr i32 %shl.2, 8

  %shl.3 = shl i32 %sra.0, 8
  %sra.3 = ashr i32 %shl.3, 8

  %mul1 = mul nsw i32 %sra.2, %sra.3
  ret i32 %mul1
}

; Make sure no unnecessary BFEs are emitted in the loop.
; GCN-LABEL: {{^}}mad24_destroyed_knownbits_2:
; GCN-NOT: v_bfe
; GCN: v_mad_i32_i24
; GCN-NOT: v_bfe
; GCN: v_mad_i32_i24
; GCN-NOT: v_bfe
; GCN: v_mad_i32_i24
; GCN-NOT: v_bfe
; GCN: v_mad_i32_i24
; GCN-NOT: v_bfe
define void @mad24_destroyed_knownbits_2(i32 %arg, i32 %arg1, i32 %arg2, ptr addrspace(1) %arg3) {
bb:
  br label %bb6

bb5:                                              ; preds = %bb6
  ret void

bb6:                                              ; preds = %bb6, %bb
  %tmp = phi i32 [ %tmp27, %bb6 ], [ 0, %bb ]
  %tmp7 = phi i32 [ %arg2, %bb6 ], [ 1, %bb ]
  %tmp8 = phi i32 [ %tmp26, %bb6 ], [ %arg, %bb ]
  %tmp9 = shl i32 %tmp7, 8
  %tmp10 = ashr exact i32 %tmp9, 8
  %tmp11 = shl i32 %tmp8, 8
  %tmp12 = ashr exact i32 %tmp11, 8
  %tmp13 = mul nsw i32 %tmp12, %tmp10
  %tmp14 = add nsw i32 %tmp13, %tmp7
  %tmp15 = shl i32 %tmp14, 8
  %tmp16 = ashr exact i32 %tmp15, 8
  %tmp17 = mul nsw i32 %tmp16, %tmp10
  %tmp18 = add nsw i32 %tmp17, %tmp14
  %tmp19 = shl i32 %tmp18, 8
  %tmp20 = ashr exact i32 %tmp19, 8
  %tmp21 = mul nsw i32 %tmp20, %tmp16
  %tmp22 = add nsw i32 %tmp21, %tmp18
  %tmp23 = shl i32 %tmp22, 8
  %tmp24 = ashr exact i32 %tmp23, 8
  %tmp25 = mul nsw i32 %tmp24, %tmp20
  %tmp26 = add nsw i32 %tmp25, %tmp22
  store i32 %tmp26, ptr addrspace(1) %arg3
  %tmp27 = add nuw i32 %tmp, 1
  %tmp28 = icmp eq i32 %tmp27, %arg1
  br i1 %tmp28, label %bb5, label %bb6
}

declare i32 @llvm.amdgcn.mul.i24(i32, i32)
