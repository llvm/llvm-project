; RUN: opt -mtriple=amdgcn-- -S -structurizecfg -si-annotate-control-flow %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Uses llvm.amdgcn.break

; OPT-LABEL: @break_loop(
; OPT: bb1:
; OPT: call i64 @llvm.amdgcn.break(i64
; OPT-NEXT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT: load volatile
; OPT: xor i1 %cmp1
; OPT: call i64 @llvm.amdgcn.if.break(
; OPT: br label %Flow

; OPT: Flow:
; OPT: call i1 @llvm.amdgcn.loop(i64
; OPT: br i1 %{{[0-9]+}}, label %bb9, label %bb1

; OPT: bb9:
; OPT: call void @llvm.amdgcn.end.cf(i64

; TODO: Can remove exec fixes in return block
; GCN-LABEL: {{^}}break_loop:
; GCN: s_mov_b64 [[INITMASK:s\[[0-9]+:[0-9]+\]]], 0{{$}}

; GCN: [[LOOP_ENTRY:BB[0-9]+_[0-9]+]]: ; %bb1
; GCN: s_or_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], exec, [[INITMASK]]
; GCN: v_cmp_lt_i32_e32 vcc,
; GCN: s_and_b64 vcc, exec, vcc
; GCN-NEXT: s_cbranch_vccnz [[FLOW:BB[0-9]+_[0-9]+]]

; GCN: ; BB#2: ; %bb4
; GCN: buffer_load_dword
; GCN: v_cmp_ge_i32_e32 vcc,
; GCN: s_or_b64 [[MASK]], vcc, [[INITMASK]]

; GCN: [[FLOW]]:
; GCN: s_mov_b64 [[INITMASK]], [[MASK]]
; GCN: s_andn2_b64 exec, exec, [[MASK]]
; GCN-NEXT: s_cbranch_execnz [[LOOP_ENTRY]]

; GCN: ; BB#4: ; %bb9
; GCN-NEXT: s_or_b64 exec, exec, [[MASK]]
; GCN-NEXT: s_endpgm
define void @break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:
  %lsr.iv = phi i32 [ undef, %bb ], [ %lsr.iv.next, %bb4 ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %bb9

bb4:
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp slt i32 %tmp, %load
  br i1 %cmp1, label %bb1, label %bb9

bb9:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
