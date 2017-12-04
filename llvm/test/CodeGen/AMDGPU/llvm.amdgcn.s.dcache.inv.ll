; RUN: llc -march=amdgcn -mcpu=tahiti -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -show-mc-encoding < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare void @llvm.amdgcn.s.dcache.inv() #0
declare void @llvm.amdgcn.s.waitcnt(i32) #0

; GCN-LABEL: {{^}}test_s_dcache_inv:
; GCN-NEXT: ; %bb.0:
; SI-NEXT: s_dcache_inv ; encoding: [0x00,0x00,0xc0,0xc7]
; VI-NEXT: s_dcache_inv ; encoding: [0x00,0x00,0x80,0xc0,0x00,0x00,0x00,0x00]
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_s_dcache_inv() #0 {
  call void @llvm.amdgcn.s.dcache.inv()
  ret void
}

; GCN-LABEL: {{^}}test_s_dcache_inv_insert_wait:
; GCN-NEXT: ; %bb.0:
; GCN: s_dcache_inv
; GCN: s_waitcnt lgkmcnt(0) ; encoding
define amdgpu_kernel void @test_s_dcache_inv_insert_wait() #0 {
  call void @llvm.amdgcn.s.dcache.inv()
  call void @llvm.amdgcn.s.waitcnt(i32 127)
  br label %end

end:
  store volatile i32 3, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
