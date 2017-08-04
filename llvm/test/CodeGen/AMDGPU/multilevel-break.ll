; RUN: opt -S -mtriple=amdgcn-- -structurizecfg -si-annotate-control-flow < %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; OPT-LABEL: {{^}}define amdgpu_vs void @multi_else_break(
; OPT: main_body:
; OPT: LOOP.outer:
; OPT: LOOP:
; OPT:     [[if:%[0-9]+]] = call { i1, i64 } @llvm.amdgcn.if(
; OPT:     [[if_exec:%[0-9]+]] = extractvalue { i1, i64 } [[if]], 1
;
; OPT: Flow:
;
; Ensure two else.break calls, for both the inner and outer loops

; OPT:        call i64 @llvm.amdgcn.else.break(i64 [[if_exec]],
; OPT-NEXT:   call i64 @llvm.amdgcn.else.break(i64 [[if_exec]],
; OPT-NEXT:   call void @llvm.amdgcn.end.cf
;
; OPT: Flow1:

; GCN-LABEL: {{^}}multi_else_break:

; GCN: [[OUTER_LOOP:BB[0-9]+_[0-9]+]]: ; %LOOP.outer{{$}}

; GCN: [[INNER_LOOP:BB[0-9]+_[0-9]+]]: ; %LOOP{{$}}
; GCN: s_and_saveexec_b64 [[SAVE_BREAK:s\[[0-9]+:[0-9]+\]]], vcc

; GCN: BB{{[0-9]+}}_{{[0-9]+}}: ; %Flow{{$}}
; GCN-NEXT: ; in Loop: Header=[[INNER_LOOP]] Depth=2

; Ensure extra or eliminated
; GCN-NEXT: s_or_b64 exec, exec, [[SAVE_BREAK]]
; GCN-NEXT: s_or_b64 [[OR_BREAK:s\[[0-9]+:[0-9]+\]]], [[SAVE_BREAK]], s{{\[[0-9]+:[0-9]+\]}}
; GCN-NEXT: s_andn2_b64 exec, exec, [[OR_BREAK]]
; GCN-NEXT: s_cbranch_execnz [[INNER_LOOP]]

; GCN: ; BB#{{[0-9]+}}: ; %Flow1{{$}}
; GCN-NEXT: ; in Loop: Header=[[OUTER_LOOP]] Depth=1

; Ensure copy is eliminated
; GCN-NEXT: s_or_b64 exec, exec, [[OR_BREAK]]
; GCN-NEXT: s_or_b64 [[OUTER_OR_BREAK:s\[[0-9]+:[0-9]+\]]], [[SAVE_BREAK]], s{{\[[0-9]+:[0-9]+\]}}
; GCN-NEXT: s_andn2_b64 exec, exec, [[OUTER_OR_BREAK]]
; GCN-NEXT: s_cbranch_execnz [[OUTER_LOOP]]
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
; OPT: llvm.amdgcn.break
; OPT: llvm.amdgcn.loop
; OPT: llvm.amdgcn.if.break
; OPT: llvm.amdgcn.if.break
; OPT: llvm.amdgcn.end.cf

; GCN-LABEL: {{^}}multi_if_break_loop:
; GCN: s_mov_b64 [[BREAK_REG:s\[[0-9]+:[0-9]+\]]], 0{{$}}

; GCN: [[LOOP:BB[0-9]+_[0-9]+]]: ; %bb1{{$}}

; Uses a copy intsead of an or
; GCN: s_mov_b64 [[COPY:s\[[0-9]+:[0-9]+\]]], [[BREAK_REG]]
; GCN: s_or_b64 [[BREAK_REG]], exec, [[COPY]]
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
