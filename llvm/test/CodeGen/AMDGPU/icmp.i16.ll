; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s| FileCheck -check-prefix=GCN -check-prefix=SI %s

;;;==========================================================================;;;
;; 16-bit integer comparisons
;;;==========================================================================;;;

; GCN-LABEL: {{^}}i16_eq:
; VI: v_cmp_eq_u16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_eq_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_eq(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp eq i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ne:
; VI: v_cmp_ne_u16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_ne_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ne(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp ne i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ugt:
; VI: v_cmp_gt_u16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_gt_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ugt(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp ugt i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_uge:
; VI: v_cmp_ge_u16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_ge_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_uge(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp uge i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ult:
; VI: v_cmp_lt_u16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_lt_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ult(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp ult i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ule:
; VI: v_cmp_le_u16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_le_u32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ule(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp ule i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void

}

; GCN-LABEL: {{^}}i16_sgt:
; VI: v_cmp_gt_i16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_gt_i32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_sgt(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp sgt i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_sge:
; VI: v_cmp_ge_i16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_ge_i32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_sge(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp sge i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_slt:
; VI: v_cmp_lt_i16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_lt_i32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_slt(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp slt i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_sle:
; VI: v_cmp_le_i16_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_le_i32_e32 vcc, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_sle(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %b.gep = getelementptr inbounds i16, ptr addrspace(1) %b.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %b = load i16, ptr addrspace(1) %b.gep
  %tmp0 = icmp sle i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; These should be commuted to reduce code size
; GCN-LABEL: {{^}}i16_eq_v_s:
; VI: v_cmp_eq_u16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_eq_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_eq_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp eq i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ne_v_s:
; VI: v_cmp_ne_u16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_ne_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ne_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp ne i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ugt_v_s:
; VI: v_cmp_lt_u16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_lt_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ugt_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp ugt i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_uge_v_s:
; VI: v_cmp_le_u16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_le_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_uge_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp uge i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ult_v_s:
; VI: v_cmp_gt_u16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_gt_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ult_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp ult i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_ule_v_s:
; VI: v_cmp_ge_u16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_ge_u32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_ule_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp ule i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_sgt_v_s:
; VI: v_cmp_lt_i16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_lt_i32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_sgt_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp sgt i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_sge_v_s:
; VI: v_cmp_le_i16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_le_i32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_sge_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp sge i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_slt_v_s:
; VI: v_cmp_gt_i16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_gt_i32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_slt_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp slt i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}i16_sle_v_s:
; VI: v_cmp_ge_i16_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp_ge_i32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @i16_sle_v_s(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, i16 %b) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %a.gep = getelementptr inbounds i16, ptr addrspace(1) %a.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %a = load i16, ptr addrspace(1) %a.gep
  %tmp0 = icmp sle i16 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, ptr addrspace(1) %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
