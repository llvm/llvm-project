; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=CM -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_smul24_i32:
; GCN: s_mul_i32

; Signed 24-bit multiply is not supported on pre-Cayman GPUs.
; EG: MULLO_INT

; CM: MULLO_INT
define amdgpu_kernel void @test_smul24_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
entry:
  %a.shl = shl i32 %a, 8
  %a.24 = ashr i32 %a.shl, 8
  %b.shl = shl i32 %b, 8
  %b.24 = ashr i32 %b.shl, 8
  %mul24 = mul i32 %a.24, %b.24
  store i32 %mul24, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_smulhi24_i64:
; GCN-NOT: bfe
; GCN-NOT: ashr
; GCN: v_mul_hi_i32_i24_e32 [[RESULT:v[0-9]+]],
; GCN-NEXT: buffer_store_dword [[RESULT]]

; EG: ASHR
; EG: ASHR
; EG: MULHI_INT

; CM-NOT: ASHR
; CM: MULHI_INT24
; CM: MULHI_INT24
; CM: MULHI_INT24
; CM: MULHI_INT24
define amdgpu_kernel void @test_smulhi24_i64(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
entry:
  %a.shl = shl i32 %a, 8
  %a.24 = ashr i32 %a.shl, 8
  %b.shl = shl i32 %b, 8
  %b.24 = ashr i32 %b.shl, 8
  %a.24.i64 = sext i32 %a.24 to i64
  %b.24.i64 = sext i32 %b.24 to i64
  %mul48 = mul i64 %a.24.i64, %b.24.i64
  %mul48.hi = lshr i64 %mul48, 32
  %mul24hi = trunc i64 %mul48.hi to i32
  store i32 %mul24hi, i32 addrspace(1)* %out
  ret void
}

; This requires handling of the original 64-bit mul node to eliminate
; unnecessary extension instructions because after legalization they
; will not be removed by SimplifyDemandedBits because there are
; multiple uses by the separate mul and mulhi.

; FUNC-LABEL: {{^}}test_smul24_i64:
; GCN: s_load_dword s
; GCN: s_load_dword s

; GCN-NOT: ashr

; GCN-DAG: v_mul_hi_i32_i24_e32
; GCN-DAG: s_mul_i32

; GCN: buffer_store_dwordx2
define amdgpu_kernel void @test_smul24_i64(i64 addrspace(1)* %out, [8 x i32], i32 %a, [8 x i32], i32 %b) #0 {
  %shl.i = shl i32 %a, 8
  %shr.i = ashr i32 %shl.i, 8
  %conv.i = sext i32 %shr.i to i64
  %shl1.i = shl i32 %b, 8
  %shr2.i = ashr i32 %shl1.i, 8
  %conv3.i = sext i32 %shr2.i to i64
  %mul.i = mul i64 %conv3.i, %conv.i
  store i64 %mul.i, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_smul24_i64_square:
; GCN: s_load_dword [[A:s[0-9]+]]
; GCN-DAG: v_mul_hi_i32_i24_e64 v{{[0-9]+}}, [[A]], [[A]]
; GCN-DAG: s_mul_i32 s{{[0-9]+}}, [[A]], [[A]]
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @test_smul24_i64_square(i64 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %shl.i = shl i32 %a, 8
  %shr.i = ashr i32 %shl.i, 8
  %conv.i = sext i32 %shr.i to i64
  %mul.i = mul i64 %conv.i, %conv.i
  store i64 %mul.i, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_smul24_i33:
; GCN: s_load_dword s
; GCN: s_load_dword s

; GCN-NOT: and
; GCN-NOT: lshr

; GCN-DAG: s_mul_i32
; GCN-DAG: v_mul_hi_i32_i24_e32
; SI: v_lshl_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, 31
; SI: v_ashr_i64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, 31

; VI: v_lshlrev_b64 v{{\[[0-9]+:[0-9]+\]}}, 31, v{{\[[0-9]+:[0-9]+\]}}
; VI: v_ashrrev_i64 v{{\[[0-9]+:[0-9]+\]}}, 31, v{{\[[0-9]+:[0-9]+\]}}

; GCN: buffer_store_dwordx2
define amdgpu_kernel void @test_smul24_i33(i64 addrspace(1)* %out, i33 %a, i33 %b) #0 {
entry:
  %a.shl = shl i33 %a, 9
  %a.24 = ashr i33 %a.shl, 9
  %b.shl = shl i33 %b, 9
  %b.24 = ashr i33 %b.shl, 9
  %mul24 = mul i33 %a.24, %b.24
  %ext = sext i33 %mul24 to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_smulhi24_i33:
; SI: s_load_dword s
; SI: s_load_dword s

; SI-NOT: bfe

; SI: v_mul_hi_i32_i24_e32 v[[MUL_HI:[0-9]+]],
; SI-NEXT: v_and_b32_e32 v[[HI:[0-9]+]], 1, v[[MUL_HI]]
; SI-NEXT: buffer_store_dword v[[HI]]
define amdgpu_kernel void @test_smulhi24_i33(i32 addrspace(1)* %out, i33 %a, i33 %b) {
entry:
  %tmp0 = shl i33 %a, 9
  %a_24 = ashr i33 %tmp0, 9
  %tmp1 = shl i33 %b, 9
  %b_24 = ashr i33 %tmp1, 9
  %tmp2 = mul i33 %a_24, %b_24
  %hi = lshr i33 %tmp2, 32
  %trunc = trunc i33 %hi to i32

  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}simplify_i24_crash:
; GCN: s_mul_i32 s[[VAL:[0-9]+]]
; GCN: v_mov_b32_e32 v[[VAL_LO:[0-9]+]], s[[VAL]]
; GCN: v_mov_b32_e32 v[[VAL_HI:[0-9]+]], s[[VAL]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VAL_LO]]:[[VAL_HI]]{{\]}}
define amdgpu_kernel void @simplify_i24_crash(<2 x i32> addrspace(1)* %out, i32 %arg0, <2 x i32> %arg1, <2 x i32> %arg2) {
bb:
  %cmp = icmp eq i32 %arg0, 0
  br i1 %cmp, label %bb11, label %bb7

bb11:
  %tmp14 = shufflevector <2 x i32> %arg1, <2 x i32> undef, <2 x i32> zeroinitializer
  %tmp16 = shufflevector <2 x i32> %arg2, <2 x i32> undef, <2 x i32> zeroinitializer
  %tmp17 = shl <2 x i32> %tmp14, <i32 8, i32 8>
  %tmp18 = ashr <2 x i32> %tmp17, <i32 8, i32 8>
  %tmp19 = shl <2 x i32> %tmp16, <i32 8, i32 8>
  %tmp20 = ashr <2 x i32> %tmp19, <i32 8, i32 8>
  %tmp21 = mul <2 x i32> %tmp18, %tmp20
  store <2 x i32> %tmp21, <2 x i32> addrspace(1)* %out
  br label %bb7

bb7:
  ret void

}
attributes #0 = { nounwind }
