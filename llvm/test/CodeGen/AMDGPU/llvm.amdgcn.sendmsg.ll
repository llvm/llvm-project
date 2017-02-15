;RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
;RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}test_interrupt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_INTERRUPT)
define void @test_interrupt() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 0);
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
define void @test_gs_emit() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 34, i32 0);
  ret void
}

; GCN-LABEL: {{^}}test_gs_cut:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1)
define void @test_gs_cut() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 274, i32 0);
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit_cut:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define void @test_gs_emit_cut() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 562, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_done:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
define void @test_gs_done() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}sendmsg:
; GCN: s_mov_b32 m0, s0
; VI-NEXT: s_nop 0
; GCN-NEXT: sendmsg(MSG_GS_DONE, GS_OP_NOP)
; GCN-NEXT: s_endpgm
define amdgpu_gs void @sendmsg(i32 inreg %a) #0 {
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 %a)
  ret void
}

; GCN-LABEL: {{^}}sendmsghalt:
; GCN: s_mov_b32 m0, s0
; VI-NEXT: s_nop 0
; GCN-NEXT: s_sendmsghalt sendmsg(MSG_INTERRUPT)
; GCN-NEXT: s_endpgm
define void @sendmsghalt(i32 inreg %a) #0 {
  call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 %a)
  ret void
}

; GCN-LABEL: {{^}}test_interrupt_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_INTERRUPT)
define void @test_interrupt_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 0)
define void @test_gs_emit_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 34, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_cut_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_CUT, 1)
define void @test_gs_cut_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 274, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit_cut_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define void @test_gs_emit_cut_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 562, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_done_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS_DONE, GS_OP_NOP)
define void @test_gs_done_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 3, i32 0)
  ret void
}

; Legacy
; GCN-LABEL: {{^}}test_legacy_interrupt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_INTERRUPT)
define void @test_legacy_interrupt() {
body:
  call void @llvm.SI.sendmsg(i32 1, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_legacy_gs_emit:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
define void @test_legacy_gs_emit() {
body:
  call void @llvm.SI.sendmsg(i32 34, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_legacy_gs_cut:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1)
define void @test_legacy_gs_cut() {
body:
  call void @llvm.SI.sendmsg(i32 274, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_legacy_gs_emit_cut:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define void @test_legacy_gs_emit_cut() {
body:
  call void @llvm.SI.sendmsg(i32 562, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_legacy_gs_done:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
define void @test_legacy_gs_done() {
body:
  call void @llvm.SI.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}sendmsg_legacy:
; GCN: s_mov_b32 m0, s0
; VI-NEXT: s_nop 0
; GCN-NEXT: sendmsg(MSG_GS_DONE, GS_OP_NOP)
; GCN-NEXT: s_endpgm
define amdgpu_gs void @sendmsg_legacy(i32 inreg %a) #0 {
  call void @llvm.SI.sendmsg(i32 3, i32 %a)
  ret void
}

declare void @llvm.amdgcn.s.sendmsg(i32, i32) #0
declare void @llvm.amdgcn.s.sendmsghalt(i32, i32) #0
declare void @llvm.SI.sendmsg(i32, i32) #0

attributes #0 = { nounwind }
