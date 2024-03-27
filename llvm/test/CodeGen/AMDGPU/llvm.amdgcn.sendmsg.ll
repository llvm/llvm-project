; XFAIL: *
;RUN: llc -mtriple=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,SIVI %s
;RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,VIPLUS,SIVI %s
;RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,VIPLUS,GFX9 %s

; GCN-LABEL: {{^}}test_interrupt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_INTERRUPT)
define amdgpu_kernel void @test_interrupt() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 0);
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
define amdgpu_kernel void @test_gs_emit() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 34, i32 0);
  ret void
}

; GCN-LABEL: {{^}}test_gs_cut:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1)
define amdgpu_kernel void @test_gs_cut() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 274, i32 0);
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit_cut:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define amdgpu_kernel void @test_gs_emit_cut() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 562, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_done:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
define amdgpu_kernel void @test_gs_done() {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_alloc_req:
; GCN: s_mov_b32 m0, s0
; GCN-NOT: s_mov_b32 m0
; VIPLUS-NEXT: s_nop 0
; SIVI: s_sendmsg sendmsg(9, 0, 0)
; GFX9: s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
define amdgpu_kernel void @test_gs_alloc_req(i32 inreg %a) {
body:
  call void @llvm.amdgcn.s.sendmsg(i32 9, i32 %a)
  ret void
}

; GCN-LABEL: {{^}}sendmsg:
; GCN: s_mov_b32 m0, s0
; VIPLUS-NEXT: s_nop 0
; GCN-NEXT: sendmsg(MSG_GS_DONE, GS_OP_NOP)
; GCN-NEXT: s_endpgm
define amdgpu_gs void @sendmsg(i32 inreg %a) #0 {
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 %a)
  ret void
}

; GCN-LABEL: {{^}}sendmsghalt:
; GCN: s_mov_b32 m0, s0
; VIPLUS-NEXT: s_nop 0
; GCN-NEXT: s_sendmsghalt sendmsg(MSG_INTERRUPT)
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @sendmsghalt(i32 inreg %a) #0 {
  call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 %a)
  ret void
}

; GCN-LABEL: {{^}}test_interrupt_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_INTERRUPT)
define amdgpu_kernel void @test_interrupt_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 0)
define amdgpu_kernel void @test_gs_emit_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 34, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_cut_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_CUT, 1)
define amdgpu_kernel void @test_gs_cut_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 274, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_emit_cut_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
define amdgpu_kernel void @test_gs_emit_cut_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 562, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_gs_done_halt:
; GCN: s_mov_b32 m0, 0
; GCN-NOT: s_mov_b32 m0
; GCN: s_sendmsghalt sendmsg(MSG_GS_DONE, GS_OP_NOP)
define amdgpu_kernel void @test_gs_done_halt() {
body:
  call void @llvm.amdgcn.s.sendmsghalt(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test_mul24:
; GCN: s_and_b32 s0, s0, 0x1ff
; GCN: s_mul_i32 m0, s0, 0x3000
; GCN: s_sendmsg sendmsg(MSG_INTERRUPT)
define amdgpu_gs void @test_mul24(i32 inreg %arg) {
body:
  %tmp1 = and i32 %arg, 511
  %tmp2 = mul nuw nsw i32 %tmp1, 12288
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %tmp2)
  ret void
}

; GCN-LABEL: {{^}}if_sendmsg:
; GCN: s_cbranch_execz
; GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
define amdgpu_gs void @if_sendmsg(i32 %flag) #0 {
  %cc = icmp eq i32 %flag, 0
  br i1 %cc, label %sendmsg, label %end

sendmsg:
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  br label %end

end:
  ret void
}

declare void @llvm.amdgcn.s.sendmsg(i32, i32) #0
declare void @llvm.amdgcn.s.sendmsghalt(i32, i32) #0

attributes #0 = { nounwind }
