; RUN: llc -march=amdgcn -mcpu=bonaire -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare void @llvm.amdgcn.buffer.wbinvl1.vol() #0

; GCN-LABEL: {{^}}test_buffer_wbinvl1_vol:
; GCN-NEXT: ; %bb.0:
; CI: buffer_wbinvl1_vol ; encoding: [0x00,0x00,0xc0,0xe1,0x00,0x00,0x00,0x00]
; VI: buffer_wbinvl1_vol ; encoding: [0x00,0x00,0xfc,0xe0,0x00,0x00,0x00,0x00]
; GCN: _store_byte
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_buffer_wbinvl1_vol(ptr addrspace(1) %ptr) #0 {
  call void @llvm.amdgcn.buffer.wbinvl1.vol()
; This used to crash in hazard recognizer
  store i8 0, ptr addrspace(1) %ptr, align 1
  ret void
}

attributes #0 = { nounwind }
