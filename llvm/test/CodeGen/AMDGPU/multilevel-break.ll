; RUN: opt -S -mtriple=amdgcn-- -structurizecfg -si-annotate-control-flow < %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; OPT-LABEL: {{^}}define amdgpu_vs void @multi_else_break(
; OPT: main_body:
; OPT: LOOP.outer:
; OPT: LOOP:
; OPT:     [[if:%[0-9]+]] = call { i1, i64 } @llvm.amdgcn.if.i64(
; OPT:     [[if_exec:%[0-9]+]] = extractvalue { i1, i64 } [[if]], 1
;
; OPT: Flow:
;
; Ensure two if.break calls, for both the inner and outer loops

; OPT:        call void @llvm.amdgcn.end.cf
; OPT-NEXT:   call i64 @llvm.amdgcn.if.break.i64.i64(i1
; OPT-NEXT:   call i1 @llvm.amdgcn.loop.i64(i64
; OPT-NEXT:   call i64 @llvm.amdgcn.if.break.i64.i64(i1
;
; OPT: Flow1:

; GCN-LABEL: {{^}}multi_else_break:

; GCN: ; %main_body
; GCN:      s_mov_b64           [[LEFT_OUTER:s\[[0-9]+:[0-9]+\]]], 0{{$}}

; GCN: [[FLOW2:BB[0-9]+_[0-9]+]]: ; %Flow2
; GCN:      s_or_b64            exec, exec, [[LEFT_INNER:s\[[0-9]+:[0-9]+\]]]
; GCN:      s_and_b64           [[TMP1:s\[[0-9]+:[0-9]+\]]], exec, [[BREAK_OUTER:s\[[0-9]+:[0-9]+\]]]
; GCN:      s_or_b64            [[LEFT_OUTER:s\[[0-9]+:[0-9]+\]]], [[TMP1]], [[LEFT_OUTER]]
; GCN:      s_andn2_b64         exec, exec, [[LEFT_OUTER]]
; GCN:      s_cbranch_execz    [[IF_BLOCK:BB[0-9]+_[0-9]+]]

; GCN: [[OUTER_LOOP:BB[0-9]+_[0-9]+]]: ; %LOOP.outer{{$}}
; GCN:      s_mov_b64           [[LEFT_INNER]], 0{{$}}

; GCN: ; %Flow
; GCN:      s_or_b64            exec, exec, [[SAVE_EXEC:s\[[0-9]+:[0-9]+\]]]
; GCN:      s_and_b64           [[TMP0:s\[[0-9]+:[0-9]+\]]], exec, [[BREAK_INNER:s\[[0-9]+:[0-9]+\]]]
; GCN:      s_or_b64            [[LEFT_INNER]], [[TMP0]], [[LEFT_INNER]]
; GCN:      s_andn2_b64         exec, exec, [[LEFT_INNER]]
; GCN:      s_cbranch_execz    [[FLOW2]]

; GCN: [[INNER_LOOP:BB[0-9]+_[0-9]+]]: ; %LOOP{{$}}
; GCN:      s_and_saveexec_b64  [[SAVE_EXEC]], vcc

; FIXME: duplicate comparison
; GCN: ; %ENDIF
; GCN-DAG:  v_cmp_eq_u32_e32    vcc,
; GCN-DAG:  v_cmp_ne_u32_e64    [[TMP51NEG:s\[[0-9]+:[0-9]+\]]],

; GCN: [[IF_BLOCK]]: ; %IF
; GCN-NEXT: s_endpgm
define amdgpu_vs void @multi_else_break(<4 x float> %vec, i32 %ub, i32 %cont) {
main_body:
  br label %LOOP.outer

LOOP.outer:                                       ; preds = %ENDIF, %main_body
  %tmp43 = phi i32 [ 0, %main_body ], [ %tmp47, %ENDIF ]
  br label %LOOP

LOOP:                                             ; preds = %ENDIF, %LOOP.outer
  %tmp45 = phi i32 [ %tmp43, %LOOP.outer ], [ %tmp47, %ENDIF ]
  %tmp47 = add i32 %tmp45, 1
  %tmp48 = icmp slt i32 %tmp45, %ub
  br i1 %tmp48, label %ENDIF, label %IF

IF:                                               ; preds = %LOOP
  ret void

ENDIF:                                            ; preds = %LOOP
  %tmp51 = icmp eq i32 %tmp47, %cont
  br i1 %tmp51, label %LOOP, label %LOOP.outer
}

; OPT-LABEL: define amdgpu_kernel void @multi_if_break_loop(
; OPT: llvm.amdgcn.if.break
; OPT: llvm.amdgcn.loop
; OPT: llvm.amdgcn.if.break
; OPT: llvm.amdgcn.end.cf

; GCN-LABEL: {{^}}multi_if_break_loop:
; GCN:      s_mov_b64          [[SAVED_MASK:s\[[0-9]+:[0-9]+\]]], 0{{$}}

; GCN: [[LOOP:BB[0-9]+_[0-9]+]]: ; %Flow4
; GCN:      s_and_b64          [[ANDTMP0:s\[[0-9]+:[0-9]+\]]], exec, {{s\[[0-9]+:[0-9]+\]}}
; GCN:      s_or_b64           [[MASK1:s\[[0-9]+:[0-9]+\]]], [[ANDTMP0]], [[SAVED_MASK]]
; GCN:      s_and_b64          [[BROKEN_THREADS_MASK:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, exec
; GCN:      s_andn2_b64        exec, exec, [[MASK1]]
; GCN-NEXT: s_cbranch_execz [[LOOP_EXIT:BB[0-9]+_[0-9]+]]

; GCN: ; %bb1{{$}}
; GCN:      buffer_load_dword  [[LOAD0:v[0-9]+]],

; GCN: ; %LeafBlock1
; GCN:      v_cmp_eq_u32_e32 vcc, 1, [[LOAD0]]
; GCN:      s_and_b64 vcc, exec, vcc
; GCN:      s_cbranch_vccz [[FLOW:BB[0-9]+_[0-9]+]]

; GCN: ; %case1
; GCN:      buffer_load_dword  [[LOAD2:v[0-9]+]],
; GCN:      v_cmp_ge_i32_e32   vcc, {{v[0-9]+}}, [[LOAD2]]
; GCN:      s_orn2_b64 [[BROKEN_THREADS_MASK]], vcc, exec
; GCN:  BB1_{{[0-9]+}}:
; GCN:      s_mov_b64 [[FALSE_MASK:s\[[0-9]+:[0-9]+\]]], 0
; GCN:      s_and_b64 vcc, exec, [[FALSE_MASK]]
; GCN:      s_cbranch_vccz [[LOOP]]

; GCN: ; %LeafBlock
; GCN:     v_cmp_eq_u32_e32 vcc, 0, [[LOAD0]]
; GCN:     s_and_b64 vcc, exec, vcc
; GCN:     s_cbranch_vccz [[LOOP]]

; GCN: ; %case0
; GCN:      buffer_load_dword  [[LOAD1:v[0-9]+]],
; GCN-DAG:  v_cmp_ge_i32_e32   vcc, {{v[0-9]+}}, [[LOAD1]]
; GCN:      s_andn2_b64 [[BROKEN_THREADS_MASK]], [[BROKEN_THREADS_MASK]], exec
; GCN:      s_and_b64 [[TMP_MASK:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN:      s_or_b64 [[BROKEN_THREADS_MASK]], [[BROKEN_THREADS_MASK]], [[TMP_MASK]]
; GCN:      s_branch [[LOOP]]

; GCN: [[LOOP_EXIT]]: ; %Flow6
; GCN: 	s_or_b64 exec, exec, [[SAVED_MASK]]

define amdgpu_kernel void @multi_if_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:
  %lsr.iv = phi i32 [ undef, %bb ], [ %lsr.iv.next, %case0 ], [ %lsr.iv.next, %case1 ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  %load0 = load volatile i32, i32 addrspace(1)* undef, align 4
  switch i32 %load0, label %bb9 [
    i32 0, label %case0
    i32 1, label %case1
  ]

case0:
  %load1 = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp slt i32 %tmp, %load1
  br i1 %cmp1, label %bb1, label %bb9

case1:
  %load2 = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp2 = icmp slt i32 %tmp, %load2
  br i1 %cmp2, label %bb1, label %bb9

bb9:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
