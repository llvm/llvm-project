; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: opt -mtriple=amdgcn-- -S -amdgpu-unify-divergent-exit-nodes -verify %s | FileCheck -check-prefix=IR %s

; SI-LABEL: {{^}}infinite_loop:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3e7
; SI: [[LOOP:BB[0-9]+_[0-9]+]]:  ; %loop
; SI: s_waitcnt lgkmcnt(0)
; SI: buffer_store_dword [[REG]]
; SI: s_branch [[LOOP]]
define amdgpu_kernel void @infinite_loop(i32 addrspace(1)* %out) {
entry:
  br label %loop

loop:
  store volatile i32 999, i32 addrspace(1)* %out, align 4
  br label %loop
}


; IR-LABEL: @infinite_loop_ret(
; IR:  br i1 %cond, label %loop, label %UnifiedReturnBlock

; IR: loop:
; IR: store volatile i32 999, i32 addrspace(1)* %out, align 4
; IR: br i1 true, label %loop, label %UnifiedReturnBlock

; IR: UnifiedReturnBlock:
; IR:  ret void


; SI-LABEL: {{^}}infinite_loop_ret:
; SI: s_cbranch_execz [[RET:BB[0-9]+_[0-9]+]]

; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3e7
; SI: [[LOOP:BB[0-9]+_[0-9]+]]:  ; %loop
; SI: s_and_b64 vcc, exec, -1
; SI: s_waitcnt lgkmcnt(0)
; SI: buffer_store_dword [[REG]]
; SI: s_cbranch_vccnz [[LOOP]]

; SI: [[RET]]:  ; %UnifiedReturnBlock
; SI: s_endpgm
define amdgpu_kernel void @infinite_loop_ret(i32 addrspace(1)* %out) {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %tmp, 1
  br i1 %cond, label %loop, label %return

loop:
  store volatile i32 999, i32 addrspace(1)* %out, align 4
  br label %loop

return:
  ret void
}


; IR-LABEL: @infinite_loops(
; IR: br i1 undef, label %loop1, label %loop2

; IR: loop1:
; IR: store volatile i32 999, i32 addrspace(1)* %out, align 4
; IR: br i1 true, label %loop1, label %DummyReturnBlock

; IR: loop2:
; IR: store volatile i32 888, i32 addrspace(1)* %out, align 4
; IR: br i1 true, label %loop2, label %DummyReturnBlock

; IR: DummyReturnBlock:
; IR: ret void


; SI-LABEL: {{^}}infinite_loops:

; SI: v_mov_b32_e32 [[REG1:v[0-9]+]], 0x3e7
; SI: s_and_b64 vcc, exec, -1

; SI: [[LOOP1:BB[0-9]+_[0-9]+]]:  ; %loop1
; SI: s_waitcnt lgkmcnt(0)
; SI: buffer_store_dword [[REG1]]
; SI: s_cbranch_vccnz [[LOOP1]]
; SI: s_branch [[RET:BB[0-9]+_[0-9]+]]

; SI: v_mov_b32_e32 [[REG2:v[0-9]+]], 0x378
; SI: s_and_b64 vcc, exec, -1

; SI: [[LOOP2:BB[0-9]+_[0-9]+]]:  ; %loop2
; SI: s_waitcnt lgkmcnt(0)
; SI: buffer_store_dword [[REG2]]
; SI: s_cbranch_vccnz [[LOOP2]]

; SI: [[RET]]:  ; %DummyReturnBlock
; SI: s_endpgm
define amdgpu_kernel void @infinite_loops(i32 addrspace(1)* %out) {
entry:
  br i1 undef, label %loop1, label %loop2

loop1:
  store volatile i32 999, i32 addrspace(1)* %out, align 4
  br label %loop1

loop2:
  store volatile i32 888, i32 addrspace(1)* %out, align 4
  br label %loop2
}



; IR-LABEL: @infinite_loop_nest_ret(
; IR: br i1 %cond1, label %outer_loop, label %UnifiedReturnBlock

; IR: outer_loop:
; IR: br label %inner_loop

; IR: inner_loop:
; IR: store volatile i32 999, i32 addrspace(1)* %out, align 4
; IR: %cond3 = icmp eq i32 %tmp, 3
; IR: br i1 true, label %TransitionBlock, label %UnifiedReturnBlock

; IR: TransitionBlock:
; IR: br i1 %cond3, label %inner_loop, label %outer_loop

; IR: UnifiedReturnBlock:
; IR: ret void

; SI-LABEL: {{^}}infinite_loop_nest_ret:
; SI: s_cbranch_execz [[RET:BB[0-9]+_[0-9]+]]

; SI: s_mov_b32
; SI: [[OUTER_LOOP:BB[0-9]+_[0-9]+]]:  ; %outer_loop

; SI: [[INNER_LOOP:BB[0-9]+_[0-9]+]]:  ; %inner_loop
; SI: s_waitcnt expcnt(0)
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3e7
; SI: s_waitcnt lgkmcnt(0)
; SI: buffer_store_dword [[REG]]

; SI: s_andn2_b64 exec
; SI: s_cbranch_execnz [[INNER_LOOP]]

; SI: s_andn2_b64 exec
; SI: s_cbranch_execnz [[OUTER_LOOP]]

; SI: [[RET]]:  ; %UnifiedReturnBlock
; SI: s_endpgm
define amdgpu_kernel void @infinite_loop_nest_ret(i32 addrspace(1)* %out) {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %cond1 = icmp eq i32 %tmp, 1
  br i1 %cond1, label %outer_loop, label %return

outer_loop:
 ; %cond2 = icmp eq i32 %tmp, 2
 ; br i1 %cond2, label %outer_loop, label %inner_loop
 br label %inner_loop

inner_loop:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 999, i32 addrspace(1)* %out, align 4
  %cond3 = icmp eq i32 %tmp, 3
  br i1 %cond3, label %inner_loop, label %outer_loop

return:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
