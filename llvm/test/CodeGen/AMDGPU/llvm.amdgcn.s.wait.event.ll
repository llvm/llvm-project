; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_wait_event_export_ready:
; GCN: s_wait_event 0x2
define amdgpu_ps void @test_wait_event_export_ready() {
entry:
  call void @llvm.amdgcn.s.wait.event.export.ready()
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_0:
; GCN: s_wait_event 0x0
define amdgpu_ps void @test_wait_event_0() {
  call void @llvm.amdgcn.s.wait.event(i16 0)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_1:
; GCN: s_wait_event 0x1
define amdgpu_ps void @test_wait_event_1() {
  call void @llvm.amdgcn.s.wait.event(i16 1)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_2:
; GCN: s_wait_event 0x2
define amdgpu_ps void @test_wait_event_2() {
  call void @llvm.amdgcn.s.wait.event(i16 2)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_3:
; GCN: s_wait_event 0x3
define amdgpu_ps void @test_wait_event_3() {
  call void @llvm.amdgcn.s.wait.event(i16 3)
  ret void
}

; GCN-LABEL: {{^}}test_wait_event_max:
; GCN: s_wait_event 0xffff
define amdgpu_ps void @test_wait_event_max() {
  call void @llvm.amdgcn.s.wait.event(i16 -1)
  ret void
}
