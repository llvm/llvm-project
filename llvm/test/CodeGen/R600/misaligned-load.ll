; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI: @byte_aligned_load64
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: s_endpgm
define void @byte_aligned_load64(i64 addrspace(1)* %out, i64 addrspace(3)* %in) {
entry:
  %0 = load i64 addrspace(3)* %in, align 1
  store i64 %0, i64 addrspace(1)* %out
  ret void
}
