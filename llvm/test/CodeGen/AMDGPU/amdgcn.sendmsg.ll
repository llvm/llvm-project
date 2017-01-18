;RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
;RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test_interrupt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_INTERRUPT)
define void @test_interrupt() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 0);
  ret void
}

; CHECK-LABEL: {{^}}test_gs_emit:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
define void @test_gs_emit() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 34, i32 0);
  ret void
}

; CHECK-LABEL: {{^}}test_gs_cut:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1)
define void @test_gs_cut() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 274, i32 0);
  ret void
}

; CHECK-LABEL: {{^}}test_gs_emit_cut:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define void @test_gs_emit_cut() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 562, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_gs_done:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
define void @test_gs_done() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}


; CHECK-LABEL: {{^}}test_interrupt_halt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsghalt sendmsg(MSG_INTERRUPT)
define void @test_interrupt_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_gs_emit_halt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 0)
define void @test_gs_emit_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 34, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_gs_cut_halt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsghalt sendmsg(MSG_GS, GS_OP_CUT, 1)
define void @test_gs_cut_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 274, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_gs_emit_cut_halt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define void @test_gs_emit_cut_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 562, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_gs_done_halt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsghalt sendmsg(MSG_GS_DONE, GS_OP_NOP)
define void @test_gs_done_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 3, i32 0)
  ret void
}

; Legacy
; CHECK-LABEL: {{^}}test_legacy_interrupt:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_INTERRUPT)
define void @test_legacy_interrupt() {
body:
  call void @llvm.SI.sendmsg(i32 1, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_legacy_gs_emit:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
define void @test_legacy_gs_emit() {
body:
  call void @llvm.SI.sendmsg(i32 34, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_legacy_gs_cut:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1)
define void @test_legacy_gs_cut() {
body:
  call void @llvm.SI.sendmsg(i32 274, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_legacy_gs_emit_cut:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define void @test_legacy_gs_emit_cut() {
body:
  call void @llvm.SI.sendmsg(i32 562, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test_legacy_gs_done:
; CHECK: s_mov_b32 m0, 0
; CHECK-NOT: s_mov_b32 m0
; CHECK: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
define void @test_legacy_gs_done() {
body:
  call void @llvm.SI.sendmsg(i32 3, i32 0)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.amdgcn.s.sendmsg(i32, i32) #0
declare void @llvm.amdgcn.s.sendmsghalt(i32, i32) #0
declare void @llvm.SI.sendmsg(i32, i32) #0

attributes #0 = { nounwind }
