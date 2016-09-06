;RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs| FileCheck --check-prefix=SI --check-prefix=FUNC %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs| FileCheck --check-prefix=SI --check-prefix=FUNC %s

; XXX: Merge this into setcc, once R600 supports 64-bit operations

;;;==========================================================================;;;
;; Double comparisons
;;;==========================================================================;;;

; FUNC-LABEL: {{^}}f64_oeq:
; SI: v_cmp_eq_f64
define void @f64_oeq(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp oeq double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ogt:
; SI: v_cmp_gt_f64
define void @f64_ogt(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ogt double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_oge:
; SI: v_cmp_ge_f64
define void @f64_oge(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp oge double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_olt:
; SI: v_cmp_lt_f64
define void @f64_olt(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp olt double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ole:
; SI: v_cmp_le_f64
define void @f64_ole(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ole double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_one:
; SI: v_cmp_lg_f64_e32 vcc
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define void @f64_one(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp one double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ord:
; SI: v_cmp_o_f64
define void @f64_ord(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ord double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ueq:
; SI: v_cmp_nlg_f64_e32 vcc
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define void @f64_ueq(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ueq double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ugt:

; SI: v_cmp_nle_f64_e32 vcc
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define void @f64_ugt(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ugt double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_uge:
; SI: v_cmp_nlt_f64_e32 vcc
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define void @f64_uge(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp uge double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ult:
; SI: v_cmp_nge_f64_e32 vcc
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define void @f64_ult(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ult double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_ule:
; SI: v_cmp_ngt_f64_e32 vcc
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define void @f64_ule(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp ule double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_une:
; SI: v_cmp_neq_f64
define void @f64_une(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp une double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f64_uno:
; SI: v_cmp_u_f64
define void @f64_uno(i32 addrspace(1)* %out, double %a, double %b) {
entry:
  %0 = fcmp uno double %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

;;;==========================================================================;;;
;; 64-bit integer comparisons
;;;==========================================================================;;;

; FUNC-LABEL: {{^}}i64_eq:
; SI: v_cmp_eq_i64
define void @i64_eq(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp eq i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_ne:
; SI: v_cmp_ne_i64
define void @i64_ne(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp ne i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_ugt:
; SI: v_cmp_gt_u64
define void @i64_ugt(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp ugt i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_uge:
; SI: v_cmp_ge_u64
define void @i64_uge(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp uge i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_ult:
; SI: v_cmp_lt_u64
define void @i64_ult(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp ult i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_ule:
; SI: v_cmp_le_u64
define void @i64_ule(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp ule i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_sgt:
; SI: v_cmp_gt_i64
define void @i64_sgt(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp sgt i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_sge:
; SI: v_cmp_ge_i64
define void @i64_sge(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp sge i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_slt:
; SI: v_cmp_lt_i64
define void @i64_slt(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp slt i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i64_sle:
; SI: v_cmp_le_i64
define void @i64_sle(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %0 = icmp sle i64 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
