; RUN: llc -mtriple=amdgcn -mcpu=fiji -show-mc-encoding < %s | FileCheck -check-prefix=VI %s

declare void @llvm.amdgcn.s.dcache.wb.vol() nounwind
declare void @llvm.amdgcn.s.waitcnt(i32) nounwind

; VI-LABEL: {{^}}test_s_dcache_wb_vol:
; VI-NEXT: ; %bb.0:
; VI-NEXT: s_dcache_wb_vol ; encoding: [0x00,0x00,0x8c,0xc0,0x00,0x00,0x00,0x00]
; VI-NEXT: s_endpgm
define amdgpu_kernel void @test_s_dcache_wb_vol() nounwind {
  call void @llvm.amdgcn.s.dcache.wb.vol()
  ret void
}

; VI-LABEL: {{^}}test_s_dcache_wb_vol_insert_wait:
; VI-NEXT: ; %bb.0:
; VI: s_dcache_wb_vol
; VI: s_waitcnt lgkmcnt(0) ; encoding
define amdgpu_kernel void @test_s_dcache_wb_vol_insert_wait() nounwind {
  call void @llvm.amdgcn.s.dcache.wb.vol()
  call void @llvm.amdgcn.s.waitcnt(i32 127)
  br label %end

end:
  store volatile i32 3, ptr addrspace(1) undef
  ret void
}
