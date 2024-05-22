; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=redwood -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=R600 -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FUNC-LABEL: {{^}}setcc_v2i32:
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW]}}, KC0[3].X, KC0[3].Z
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW]}}, KC0[2].W, KC0[3].Y

; GCN: s_cmp_eq_u32
; GCN: s_cmp_eq_u32
define amdgpu_kernel void @setcc_v2i32(ptr addrspace(1) %out, <2 x i32> %a, <2 x i32> %b) #0 {
  %result = icmp eq <2 x i32> %a, %b
  %sext = sext <2 x i1> %result to <2 x i32>
  store <2 x i32> %sext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}setcc_v4i32:
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; GCN: s_cmp_eq_u32
; GCN: s_cmp_eq_u32
; GCN: s_cmp_eq_u32
; GCN: s_cmp_eq_u32
define amdgpu_kernel void @setcc_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %b_ptr = getelementptr <4 x i32>, ptr addrspace(1) %in, i32 1
  %a = load <4 x i32>, ptr addrspace(1) %in
  %b = load <4 x i32>, ptr addrspace(1) %b_ptr
  %result = icmp eq <4 x i32> %a, %b
  %sext = sext <4 x i1> %result to <4 x i32>
  store <4 x i32> %sext, ptr addrspace(1) %out
  ret void
}

;;;==========================================================================;;;
;; Float comparisons
;;;==========================================================================;;;

; FUNC-LABEL: {{^}}f32_oeq:
; R600: SETE_DX10
; GCN: v_cmp_eq_f32
define amdgpu_kernel void @f32_oeq(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp oeq float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ogt:
; R600: SETGT_DX10
; GCN: v_cmp_gt_f32
define amdgpu_kernel void @f32_ogt(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ogt float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_oge:
; R600: SETGE_DX10
; GCN: v_cmp_ge_f32
define amdgpu_kernel void @f32_oge(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp oge float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_olt:
; R600: SETGT_DX10
; GCN: v_cmp_lt_f32
define amdgpu_kernel void @f32_olt(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp olt float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ole:
; R600: SETGE_DX10
; GCN: v_cmp_le_f32
define amdgpu_kernel void @f32_ole(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ole float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_one:
; R600-DAG: SETGT_DX10
; R600-DAG: SETGT_DX10
; R600-DAG: OR_INT
; R600-DAG: SETNE_INT

; GCN: v_cmp_lg_f32_e32 vcc
; GCN-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f32_one(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp one float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ord:
; R600-DAG: SETE_DX10
; R600-DAG: SETE_DX10
; R600-DAG: AND_INT
; R600-DAG: SETNE_INT
; GCN: v_cmp_o_f32
define amdgpu_kernel void @f32_ord(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ord float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ueq:
; R600-DAG: SETGT_DX10
; R600-DAG: SETGT_DX10
; R600-DAG: OR_INT
; R600-DAG: SETE_INT

; GCN: v_cmp_nlg_f32_e32 vcc
; GCN-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f32_ueq(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ueq float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ugt:
; R600: SETGE
; R600: SETE_DX10
; GCN: v_cmp_nle_f32_e32 vcc
; GCN-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f32_ugt(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ugt float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_uge:
; R600: SETGT
; R600: SETE_DX10

; GCN: v_cmp_nlt_f32_e32 vcc
; GCN-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f32_uge(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp uge float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ult:
; R600: SETGE
; R600: SETE_DX10

; GCN: v_cmp_nge_f32_e32 vcc
; GCN-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f32_ult(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ult float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_ule:
; R600: SETGT
; R600: SETE_DX10

; GCN: v_cmp_ngt_f32_e32 vcc
; GCN-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f32_ule(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp ule float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_une:
; R600: SETNE_DX10
; GCN: v_cmp_neq_f32
define amdgpu_kernel void @f32_une(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp une float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}f32_uno:
; R600: SETNE_DX10
; R600: SETNE_DX10
; R600: OR_INT
; R600: SETNE_INT
; GCN: v_cmp_u_f32
define amdgpu_kernel void @f32_uno(ptr addrspace(1) %out, float %a, float %b) #0 {
entry:
  %0 = fcmp uno float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

;;;==========================================================================;;;
;; 32-bit integer comparisons
;;;==========================================================================;;;

; FUNC-LABEL: {{^}}i32_eq:
; R600: SETE_INT
; GCN: s_cmp_eq_u32
define amdgpu_kernel void @i32_eq(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp eq i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_ne:
; R600: SETNE_INT
; GCN: s_cmp_lg_u32
define amdgpu_kernel void @i32_ne(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp ne i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_ugt:
; R600: SETGT_UINT
; GCN: s_cmp_gt_u32
define amdgpu_kernel void @i32_ugt(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp ugt i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_uge:
; R600: SETGE_UINT
; GCN: s_cmp_ge_u32
define amdgpu_kernel void @i32_uge(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp uge i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_ult:
; R600: SETGT_UINT
; GCN: s_cmp_lt_u32
define amdgpu_kernel void @i32_ult(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp ult i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_ule:
; R600: SETGE_UINT
; GCN: s_cmp_le_u32
define amdgpu_kernel void @i32_ule(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp ule i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_sgt:
; R600: SETGT_INT
; GCN: s_cmp_gt_i32
define amdgpu_kernel void @i32_sgt(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp sgt i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_sge:
; R600: SETGE_INT
; GCN: s_cmp_ge_i32
define amdgpu_kernel void @i32_sge(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp sge i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_slt:
; R600: SETGT_INT
; GCN: s_cmp_lt_i32
define amdgpu_kernel void @i32_slt(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp slt i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}i32_sle:
; R600: SETGE_INT
; GCN: s_cmp_le_i32
define amdgpu_kernel void @i32_sle(ptr addrspace(1) %out, i32 %a, i32 %b) #0 {
entry:
  %0 = icmp sle i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FIXME: This does 4 compares
; FUNC-LABEL: {{^}}v3i32_eq:
; GCN-DAG: v_cmp_eq_u32
; GCN-DAG: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1,
; GCN-DAG: v_cmp_eq_u32
; GCN-DAG: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1,
; GCN-DAG: v_cmp_eq_u32
; GCN-DAG: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1,
; GCN: s_endpgm
define amdgpu_kernel void @v3i32_eq(ptr addrspace(1) %out, ptr addrspace(1) %ptra, ptr addrspace(1) %ptrb) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.a = getelementptr <3 x i32>, ptr addrspace(1) %ptra, i32 %tid
  %gep.b = getelementptr <3 x i32>, ptr addrspace(1) %ptrb, i32 %tid
  %gep.out = getelementptr <3 x i32>, ptr addrspace(1) %out, i32 %tid
  %a = load <3 x i32>, ptr addrspace(1) %gep.a
  %b = load <3 x i32>, ptr addrspace(1) %gep.b
  %cmp = icmp eq <3 x i32> %a, %b
  %ext = sext <3 x i1> %cmp to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(1) %gep.out
  ret void
}

; FUNC-LABEL: {{^}}v3i8_eq:
; GCN-DAG: v_cmp_eq_u32
; GCN-DAG: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1,
; GCN-DAG: v_cmp_eq_u32
; GCN-DAG: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1,
; GCN-DAG: v_cmp_eq_u32
; GCN-DAG: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1,
; GCN: s_endpgm
define amdgpu_kernel void @v3i8_eq(ptr addrspace(1) %out, ptr addrspace(1) %ptra, ptr addrspace(1) %ptrb) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.a = getelementptr <3 x i8>, ptr addrspace(1) %ptra, i32 %tid
  %gep.b = getelementptr <3 x i8>, ptr addrspace(1) %ptrb, i32 %tid
  %gep.out = getelementptr <3 x i8>, ptr addrspace(1) %out, i32 %tid
  %a = load <3 x i8>, ptr addrspace(1) %gep.a
  %b = load <3 x i8>, ptr addrspace(1) %gep.b
  %cmp = icmp eq <3 x i8> %a, %b
  %ext = sext <3 x i1> %cmp to <3 x i8>
  store <3 x i8> %ext, ptr addrspace(1) %gep.out
  ret void
}

; Make sure we don't try to emit i1 setcc ops
; FUNC-LABEL: setcc-i1
; GCN: s_bitcmp0_b32 s{{[0-9]+}}, 0
define amdgpu_kernel void @setcc-i1(i32 %in) #0 {
  %and = and i32 %in, 1
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %endif, label %if
if:
  unreachable
endif:
  ret void
}

; FUNC-LABEL: setcc-i1-and-xor
; GCN-DAG: v_cmp_nge_f32_e64 [[A:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 0{{$}}
; GCN-DAG: v_cmp_nle_f32_e64 [[B:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 1.0
; GCN: s_or_b64 s[2:3], [[A]], [[B]]
define amdgpu_kernel void @setcc-i1-and-xor(ptr addrspace(1) %out, float %cond) #0 {
bb0:
  %tmp5 = fcmp oge float %cond, 0.000000e+00
  %tmp7 = fcmp ole float %cond, 1.000000e+00
  %tmp9 = and i1 %tmp5, %tmp7
  %tmp11 = xor i1 %tmp9, 1
  br i1 %tmp11, label %bb2, label %bb1

bb1:
  store i32 0, ptr addrspace(1) %out
  br label %bb2

bb2:
  ret void
}

; FUNC-LABEL: setcc_v2i32_expand
; GCN: s_cmp_gt_i32
; GCN: s_cmp_gt_i32
define amdgpu_kernel void @setcc_v2i32_expand(
  ptr addrspace(1) %a,
  ptr addrspace(1) %b,
  ptr addrspace(1) %c,
  ptr addrspace(1) %r) {
entry:
  %a.val = load <2 x i32>, ptr addrspace(1) %a
  %b.val = load <2 x i32>, ptr addrspace(1) %b
  %c.val = load <2 x i32>, ptr addrspace(1) %c

  %icmp.val.1 = icmp sgt <2 x i32> %a.val, <i32 1, i32 1>
  %zext.val.1 = zext <2 x i1> %icmp.val.1 to <2 x i32>
  %shl.val.1 = shl nuw <2 x i32> %zext.val.1, <i32 31, i32 31>
  %xor.val.1 = xor <2 x i32> %shl.val.1, %b.val
  %bitcast.val.1 = bitcast <2 x i32> %xor.val.1 to <2 x float>
  %icmp.val.2 = icmp sgt <2 x i32> %c.val, <i32 1199570944, i32 1199570944>
  %select.val.1 = select <2 x i1> %icmp.val.2, <2 x float> <float 1.000000e+00, float 1.000000e+00>, <2 x float> %bitcast.val.1

  store <2 x float> %select.val.1, ptr addrspace(1) %r
  ret void
}

; FUNC-LABEL: setcc_v4i32_expand
; GCN: s_cmp_gt_i32
; GCN: s_cmp_gt_i32
; GCN: s_cmp_gt_i32
; GCN: s_cmp_gt_i32
define amdgpu_kernel void @setcc_v4i32_expand(
  ptr addrspace(1) %a,
  ptr addrspace(1) %b,
  ptr addrspace(1) %c,
  ptr addrspace(1) %r) {
entry:
  %a.val = load <4 x i32>, ptr addrspace(1) %a
  %b.val = load <4 x i32>, ptr addrspace(1) %b
  %c.val = load <4 x i32>, ptr addrspace(1) %c

  %icmp.val.1 = icmp sgt <4 x i32> %a.val, <i32 1, i32 1, i32 1, i32 1>
  %zext.val.1 = zext <4 x i1> %icmp.val.1 to <4 x i32>
  %shl.val.1 = shl nuw <4 x i32> %zext.val.1, <i32 31, i32 31, i32 31, i32 31>
  %xor.val.1 = xor <4 x i32> %shl.val.1, %b.val
  %bitcast.val.1 = bitcast <4 x i32> %xor.val.1 to <4 x float>
  %icmp.val.2 = icmp sgt <4 x i32> %c.val, <i32 1199570944, i32 1199570944, i32 1199570944, i32 1199570944>
  %select.val.1 = select <4 x i1> %icmp.val.2, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> %bitcast.val.1

  store <4 x float> %select.val.1, ptr addrspace(1) %r
  ret void
}

attributes #0 = { nounwind }
