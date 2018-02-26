; RUN: llc -march=amdgcn -mcpu=bonaire -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare void @llvm.amdgcn.s.dcache.inv.vol() #0
declare void @llvm.amdgcn.s.waitcnt(i32) #0

; GCN-LABEL: {{^}}test_s_dcache_inv_vol:
; GCN-NEXT: ; %bb.0:
; CI-NEXT: s_dcache_inv_vol ; encoding: [0x00,0x00,0x40,0xc7]
; VI-NEXT: s_dcache_inv_vol ; encoding: [0x00,0x00,0x88,0xc0,0x00,0x00,0x00,0x00]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_s_dcache_inv_vol() #0 {
  call void @llvm.amdgcn.s.dcache.inv.vol()
  ret void
}

; GCN-LABEL: {{^}}test_s_dcache_inv_vol_insert_wait:
; GCN-NEXT: ; %bb.0:
; GCN-NEXT: s_dcache_inv_vol
; GCN: s_waitcnt lgkmcnt(0) ; encoding
define amdgpu_kernel void @test_s_dcache_inv_vol_insert_wait() #0 {
  call void @llvm.amdgcn.s.dcache.inv.vol()
  call void @llvm.amdgcn.s.waitcnt(i32 127)
  br label %end

end:
  store volatile i32 3, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
