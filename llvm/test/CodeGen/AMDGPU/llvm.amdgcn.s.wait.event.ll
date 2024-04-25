; RUN: llc -global-isel=0 -mtriple=amdgcn -verify-machineinstrs -mcpu=gfx1100 < %s | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -verify-machineinstrs -mcpu=gfx1100 < %s | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -verify-machineinstrs -mcpu=gfx1200 < %s | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -verify-machineinstrs -mcpu=gfx1200 < %s | FileCheck -check-prefixes=GCN,GFX12 %s

; GCN-LABEL: {{^}}test_wait_event:
; GFX11: s_wait_event 0x0
; GFX12: s_wait_event 0x2

define amdgpu_ps void @test_wait_event() {
entry:
  call void @llvm.amdgcn.s.wait.event.export.ready()
  ret void
}
