; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.icmp.i32(i32, i32, i32) #0
declare i64 @llvm.amdgcn.icmp.i64(i64, i64, i32) #0

; GCN-LABEL: {{^}}v_icmp_i32_eq:
; GCN: v_cmp_eq_i32_e64
define void @v_icmp_i32_eq(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 32)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp:
; GCN-NOT: v_cmp_eq_i32_e64
define void @v_icmp(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 30)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i32_ne:
; GCN: v_cmp_ne_i32_e64
define void @v_icmp_i32_ne(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 33)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u32_ugt:
; GCN: v_cmp_gt_u32_e64
define void @v_icmp_u32_ugt(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 34)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u32_uge:
; GCN: v_cmp_ge_u32_e64
define void @v_icmp_u32_uge(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 35)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u32_ult:
; GCN: v_cmp_lt_u32_e64
define void @v_icmp_u32_ult(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 36)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u32_ule:
; GCN: v_cmp_le_u32_e64
define void @v_icmp_u32_ule(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 37)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_sgt:
; GCN: v_cmp_gt_i32_e64
define void @v_icmp_i32_sgt(i64 addrspace(1)* %out, i32 %src) #1 {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 38)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_sge:
; GCN: v_cmp_ge_i32_e64
define void @v_icmp_i32_sge(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 39)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_slt:
; GCN: v_cmp_lt_i32_e64
define void @v_icmp_i32_slt(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 40)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i32_sle:
; GCN: v_cmp_le_i32_e64
define void @v_icmp_i32_sle(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 41)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_eq:
; GCN: v_cmp_eq_i64_e64
define void @v_icmp_i64_eq(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 32)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_ne:
; GCN: v_cmp_ne_i64_e64
define void @v_icmp_i64_ne(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 33)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_ugt:
; GCN: v_cmp_gt_u64_e64
define void @v_icmp_u64_ugt(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 34)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_uge:
; GCN: v_cmp_ge_u64_e64
define void @v_icmp_u64_uge(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 35)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_ult:
; GCN: v_cmp_lt_u64_e64
define void @v_icmp_u64_ult(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 36)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_ule:
; GCN: v_cmp_le_u64_e64
define void @v_icmp_u64_ule(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 37)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_sgt:
; GCN: v_cmp_gt_i64_e64
define void @v_icmp_i64_sgt(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 38)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_sge:
; GCN: v_cmp_ge_i64_e64
define void @v_icmp_i64_sge(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 39)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_slt:
; GCN: v_cmp_lt_i64_e64
define void @v_icmp_i64_slt(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 40)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i64_sle:
; GCN: v_cmp_le_i64_e64
define void @v_icmp_i64_sle(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 41)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone convergent }
