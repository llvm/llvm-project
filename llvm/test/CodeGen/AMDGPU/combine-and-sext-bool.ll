; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}and_i1_sext_bool:
; GCN: v_cmp_{{gt|le}}_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_cndmask_b32_e{{32|64}} [[VAL:v[0-9]+]], 0, v{{[0-9]+}}, [[CC]]
; GCN: store_dword {{.*}}[[VAL]]
; GCN-NOT: v_cndmask_b32_e64 v{{[0-9]+}}, {{0|-1}}, {{0|-1}}
; GCN-NOT: v_and_b32_e32

define amdgpu_kernel void @and_i1_sext_bool(ptr addrspace(1) nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %x
  %v = load i32, ptr addrspace(1) %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %and = and i32 %v, %ext
  store i32 %and, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}and_sext_bool_fcmp:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_cmp_eq_f32_e32 vcc, 0, v0
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_fcmp(float %x, i32 %y) {
  %cmp = fcmp oeq float %x, 0.0
  %sext = sext i1 %cmp to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_fpclass:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_mov_b32_e32 [[K:v[0-9]+]], 0x7b
; GCN-NEXT: v_cmp_class_f32_e32 vcc, v0, [[K]]
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_fpclass(float %x, i32 %y) {
  %class = call i1 @llvm.is.fpclass(float %x, i32 123)
  %sext = sext i1 %class to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_uadd_w_overflow:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_add_i32_e32 v0, vcc, v0, v1
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_uadd_w_overflow(i32 %x, i32 %y) {
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
  %carry = extractvalue { i32, i1 } %uadd, 1
  %sext = sext i1 %carry to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_usub_w_overflow:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_sub_i32_e32 v0, vcc, v0, v1
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_usub_w_overflow(i32 %x, i32 %y) {
  %uadd = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %x, i32 %y)
  %carry = extractvalue { i32, i1 } %uadd, 1
  %sext = sext i1 %carry to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_sadd_w_overflow:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_cmp_gt_i32_e32 vcc, 0, v1
; GCN-NEXT: v_add_i32_e64 v2, s[4:5], v0, v1
; GCN-NEXT: v_cmp_lt_i32_e64 s[4:5], v2, v0
; GCN-NEXT: s_xor_b64 vcc, vcc, s[4:5]
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_sadd_w_overflow(i32 %x, i32 %y) {
  %uadd = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %carry = extractvalue { i32, i1 } %uadd, 1
  %sext = sext i1 %carry to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_ssub_w_overflow:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_cmp_gt_i32_e32 vcc, 0, v1
; GCN-NEXT: v_add_i32_e64 v2, s[4:5], v0, v1
; GCN-NEXT: v_cmp_lt_i32_e64 s[4:5], v2, v0
; GCN-NEXT: s_xor_b64 vcc, vcc, s[4:5]
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_ssub_w_overflow(i32 %x, i32 %y) {
  %uadd = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %carry = extractvalue { i32, i1 } %uadd, 1
  %sext = sext i1 %carry to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_smul_w_overflow:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_mul_hi_i32 v2, v0, v1
; GCN-NEXT: v_mul_lo_u32 v0, v0, v1
; GCN-NEXT: v_ashrrev_i32_e32 v0, 31, v0
; GCN-NEXT: v_cmp_ne_u32_e32 vcc, v2, v0
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_smul_w_overflow(i32 %x, i32 %y) {
  %uadd = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %x, i32 %y)
  %carry = extractvalue { i32, i1 } %uadd, 1
  %sext = sext i1 %carry to i32
  %and = and i32 %sext, %y
  ret i32 %and
}

; GCN-LABEL: {{^}}and_sext_bool_umul_w_overflow:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_mul_hi_u32 v0, v0, v1
; GCN-NEXT: v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NEXT: v_cndmask_b32_e32 v0, 0, v1, vcc
; GCN-NEXT: s_setpc_b64
define i32 @and_sext_bool_umul_w_overflow(i32 %x, i32 %y) {
  %uadd = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %carry = extractvalue { i32, i1 } %uadd, 1
  %sext = sext i1 %carry to i32
  %and = and i32 %sext, %y
  ret i32 %and
}


declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workitem.id.y() #0

attributes #0 = { nounwind readnone speculatable }
