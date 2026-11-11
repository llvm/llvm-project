; RUN: llc -global-isel=0 -mtriple=amdgpu12.00 < %s | FileCheck -check-prefix=GFX12 %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.00 < %s | FileCheck -check-prefix=GFX12 %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.00 -mattr=-bvh-ray-tracing-insts < %s | FileCheck -check-prefix=GFX12 %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.00 -mattr=-bvh-ray-tracing-insts < %s | FileCheck -check-prefix=GFX12 %s
; RUN: not --crash llc -mtriple=amdgpu13.10 < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.s.wait.bvhcnt

define amdgpu_ps void @test_bvhcnt() {
; GFX12-LABEL: test_bvhcnt:
; GFX12:       ; %bb.0:
; GFX12-NEXT:    s_wait_bvhcnt 0x0
; GFX12-NEXT:    s_endpgm
  call void @llvm.amdgcn.s.wait.bvhcnt(i16 0)
  ret void
}

declare void @llvm.amdgcn.s.wait.bvhcnt(i16)
