; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefixes=GCN,GFX12 %s

; GCN-LABEL: {{^}}test_wait_event_export_ready:
; GFX11: s_wait_event 0x2
; GFX12: s_wait_event { export_ready: 1 }
define amdgpu_ps void @test_wait_event_export_ready() {
entry:
  call void @llvm.amdgcn.s.wait.event.export.ready()
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_0:
; GFX11: s_wait_event { dont_wait_export_ready: 0 }
; GFX12: s_wait_event { export_ready: 0 }
define amdgpu_ps void @test_wait_event_0() {
  call void @llvm.amdgcn.s.wait.event(i16 0)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_1:
; GFX11: s_wait_event { dont_wait_export_ready: 1 }
; GFX12: s_wait_event 0x1
define amdgpu_ps void @test_wait_event_1() {
  call void @llvm.amdgcn.s.wait.event(i16 1)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_2:
; GFX11: s_wait_event 0x2
; GFX12: s_wait_event { export_ready: 1 }
define amdgpu_ps void @test_wait_event_2() {
  call void @llvm.amdgcn.s.wait.event(i16 2)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_3:
; GFX11: s_wait_event 0x3
; GFX12: s_wait_event 0x3
define amdgpu_ps void @test_wait_event_3() {
  call void @llvm.amdgcn.s.wait.event(i16 3)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_max:
; GFX11: s_wait_event 0xffff
; GFX12: s_wait_event 0xffff
define amdgpu_ps void @test_wait_event_max() {
  call void @llvm.amdgcn.s.wait.event(i16 -1)
  ret void
}
