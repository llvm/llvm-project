;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600 --check-prefix=FUNC %s
;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck --check-prefix=SI --check-prefix=FUNC %s

; FUNC-LABEL: @setcc_v2i32
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW]}}, KC0[3].X, KC0[3].Z
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW]}}, KC0[2].W, KC0[3].Y

define void @setcc_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) {
  %result = icmp eq <2 x i32> %a, %b
  %sext = sext <2 x i1> %result to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @setcc_v4i32
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @setcc_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = icmp eq <4 x i32> %a, %b
  %sext = sext <4 x i1> %result to <4 x i32>
  store <4 x i32> %sext, <4 x i32> addrspace(1)* %out
  ret void
}

;;;==========================================================================;;;
;; Float comparisons
;;;==========================================================================;;;

; FUNC-LABEL: @f32_oeq
; R600: SETE_DX10
; SI: V_CMP_EQ_F32
define void @f32_oeq(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp oeq float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ogt
; R600: SETGT_DX10
; SI: V_CMP_GT_F32
define void @f32_ogt(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ogt float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_oge
; R600: SETGE_DX10
; SI: V_CMP_GE_F32
define void @f32_oge(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp oge float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_olt
; R600: SETGT_DX10
; SI: V_CMP_LT_F32
define void @f32_olt(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp olt float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ole
; R600: SETGE_DX10
; SI: V_CMP_LE_F32
define void @f32_ole(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ole float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_one
; R600-DAG: SETE_DX10
; R600-DAG: SETE_DX10
; R600-DAG: AND_INT
; R600-DAG: SETNE_DX10
; R600-DAG: AND_INT
; R600-DAG: SETNE_INT
; SI: V_CMP_O_F32
; SI: V_CMP_NEQ_F32
; SI: S_AND_B64
define void @f32_one(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp one float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ord
; R600-DAG: SETE_DX10
; R600-DAG: SETE_DX10
; R600-DAG: AND_INT
; R600-DAG: SETNE_INT
; SI: V_CMP_O_F32
define void @f32_ord(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ord float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ueq
; R600-DAG: SETNE_DX10
; R600-DAG: SETNE_DX10
; R600-DAG: OR_INT
; R600-DAG: SETE_DX10
; R600-DAG: OR_INT
; R600-DAG: SETNE_INT
; SI: V_CMP_U_F32
; SI: V_CMP_EQ_F32
; SI: S_OR_B64
define void @f32_ueq(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ueq float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ugt
; R600: SETGE
; R600: SETE_DX10
; SI: V_CMP_U_F32
; SI: V_CMP_GT_F32
; SI: S_OR_B64
define void @f32_ugt(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ugt float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_uge
; R600: SETGT
; R600: SETE_DX10
; SI: V_CMP_U_F32
; SI: V_CMP_GE_F32
; SI: S_OR_B64
define void @f32_uge(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp uge float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ult
; R600: SETGE
; R600: SETE_DX10
; SI: V_CMP_U_F32
; SI: V_CMP_LT_F32
; SI: S_OR_B64
define void @f32_ult(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ult float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_ule
; R600: SETGT
; R600: SETE_DX10
; SI: V_CMP_U_F32
; SI: V_CMP_LE_F32
; SI: S_OR_B64
define void @f32_ule(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp ule float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_une
; R600: SETNE_DX10
; SI: V_CMP_NEQ_F32
define void @f32_une(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp une float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @f32_uno
; R600: SETNE_DX10
; R600: SETNE_DX10
; R600: OR_INT
; R600: SETNE_INT
; SI: V_CMP_U_F32
define void @f32_uno(i32 addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fcmp uno float %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

;;;==========================================================================;;;
;; 32-bit integer comparisons
;;;==========================================================================;;;

; FUNC-LABEL: @i32_eq
; R600: SETE_INT
; SI: V_CMP_EQ_I32
define void @i32_eq(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp eq i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_ne
; R600: SETNE_INT
; SI: V_CMP_NE_I32
define void @i32_ne(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp ne i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_ugt
; R600: SETGT_UINT
; SI: V_CMP_GT_U32
define void @i32_ugt(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp ugt i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_uge
; R600: SETGE_UINT
; SI: V_CMP_GE_U32
define void @i32_uge(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp uge i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_ult
; R600: SETGT_UINT
; SI: V_CMP_LT_U32
define void @i32_ult(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp ult i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_ule
; R600: SETGE_UINT
; SI: V_CMP_LE_U32
define void @i32_ule(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp ule i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_sgt
; R600: SETGT_INT
; SI: V_CMP_GT_I32
define void @i32_sgt(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp sgt i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_sge
; R600: SETGE_INT
; SI: V_CMP_GE_I32
define void @i32_sge(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp sge i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_slt
; R600: SETGT_INT
; SI: V_CMP_LT_I32
define void @i32_slt(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp slt i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @i32_sle
; R600: SETGE_INT
; SI: V_CMP_LE_I32
define void @i32_sle(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = icmp sle i32 %a, %b
  %1 = sext i1 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
