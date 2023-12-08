; RUN: llc -march=amdgcn -mcpu=tahiti -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare void @llvm.amdgcn.buffer.wbinvl1() #0

; GCN-LABEL: {{^}}test_buffer_wbinvl1:
; GCN-NEXT: ; %bb.0:
; SI-NEXT: buffer_wbinvl1 ; encoding: [0x00,0x00,0xc4,0xe1,0x00,0x00,0x00,0x00]
; VI-NEXT: buffer_wbinvl1 ; encoding: [0x00,0x00,0xf8,0xe0,0x00,0x00,0x00,0x00]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_buffer_wbinvl1() #0 {
  call void @llvm.amdgcn.buffer.wbinvl1()
  ret void
}

attributes #0 = { nounwind }
