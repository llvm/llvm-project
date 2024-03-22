; RUN: llc -mtriple=amdgcn -mcpu=tahiti -show-mc-encoding < %s | FileCheck -check-prefix=SI %s

declare void @llvm.amdgcn.buffer.wbinvl1.sc() nounwind

; SI-LABEL: {{^}}test_buffer_wbinvl1_sc:
; SI-NEXT: ; %bb.0:
; SI-NEXT: buffer_wbinvl1_sc ; encoding: [0x00,0x00,0xc0,0xe1,0x00,0x00,0x00,0x00]
; SI-NEXT: s_endpgm
define amdgpu_kernel void @test_buffer_wbinvl1_sc() nounwind {
  call void @llvm.amdgcn.buffer.wbinvl1.sc()
  ret void
}
