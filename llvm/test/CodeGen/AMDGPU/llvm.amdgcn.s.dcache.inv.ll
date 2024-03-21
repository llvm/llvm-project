; RUN: llc -mtriple=amdgcn -mcpu=tahiti -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn -mcpu=fiji -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare void @llvm.amdgcn.s.dcache.inv() nounwind
declare void @llvm.amdgcn.s.waitcnt(i32) nounwind

; GCN-LABEL: {{^}}test_s_dcache_inv:
; GCN-NEXT: ; %bb.0:
; SI-NEXT: s_dcache_inv ; encoding: [0x00,0x00,0xc0,0xc7]
; VI-NEXT: s_dcache_inv ; encoding: [0x00,0x00,0x80,0xc0,0x00,0x00,0x00,0x00]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_s_dcache_inv() nounwind {
  call void @llvm.amdgcn.s.dcache.inv()
  ret void
}

; GCN-LABEL: {{^}}test_s_dcache_inv_insert_wait:
; GCN-NEXT: ; %bb.0:
; GCN: s_dcache_inv
; GCN: s_waitcnt lgkmcnt(0) ; encoding
define amdgpu_kernel void @test_s_dcache_inv_insert_wait() nounwind {
  call void @llvm.amdgcn.s.dcache.inv()
  call void @llvm.amdgcn.s.waitcnt(i32 127)
  br label %end

end:
  store volatile i32 3, ptr addrspace(1) undef
  ret void
}
