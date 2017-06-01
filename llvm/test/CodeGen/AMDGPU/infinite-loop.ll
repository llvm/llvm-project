; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}infinite_loop:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3e7
; SI: BB0_1:
; SI: buffer_store_dword [[REG]]
; SI: s_waitcnt vmcnt(0) expcnt(0)
; SI: s_branch BB0_1
define void @infinite_loop(i32 addrspace(1)* %out) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  store i32 999, i32 addrspace(1)* %out, align 4
  br label %for.body
}

