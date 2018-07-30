;RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
;RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

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

; TODO: This should use s_mul_i32 instead of v_mul_u32_u24 + v_readfirstlane!
;
; GCN-LABEL: {{^}}test_mul24:
; GCN: v_mul_u32_u24_e32
; GCN: v_readfirstlane_b32
; GCN: s_mov_b32 m0,
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
