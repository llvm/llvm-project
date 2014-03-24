; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG %s

define void @trunc_i64_to_i32_store(i32 addrspace(1)* %out, i64 %in) {
; SI-LABEL: @trunc_i64_to_i32_store
; SI: S_LOAD_DWORD s0, s[0:1], 11
; SI: V_MOV_B32_e32 v0, s0
; SI: BUFFER_STORE_DWORD v0

; EG-LABEL: @trunc_i64_to_i32_store
; EG: MEM_RAT_CACHELESS STORE_RAW T0.X, T1.X, 1
; EG: LSHR
; EG-NEXT: 2(

  %result = trunc i64 %in to i32 store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @trunc_shl_i64:
; SI: S_LOAD_DWORDX2
; SI: S_LOAD_DWORDX2 [[SREG:s\[[0-9]+:[0-9]+\]]]
; SI: S_LSHL_B64 s{{\[}}[[LO_SREG:[0-9]+]]:{{[0-9]+\]}}, [[SREG]], 2
; SI: MOV_B32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SREG]]
; SI: BUFFER_STORE_DWORD v[[LO_VREG]],
define void @trunc_shl_i64(i32 addrspace(1)* %out, i64 %a) {
  %b = shl i64 %a, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @trunc_i32_to_i1:
; SI: V_AND_B32
; SI: V_CMP_EQ_I32
define void @trunc_i32_to_i1(i32 addrspace(1)* %out, i32 %a) {
  %trunc = trunc i32 %a to i1
  %result = select i1 %trunc, i32 1, i32 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}
