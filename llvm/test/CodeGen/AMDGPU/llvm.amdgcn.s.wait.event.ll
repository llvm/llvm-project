; RUN: llc -global-isel=0 -march=amdgcn -verify-machineinstrs -mcpu=gfx1100 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel -march=amdgcn -verify-machineinstrs -mcpu=gfx1100 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_wait_event:
; GCN: s_wait_event 0x0

define amdgpu_ps void @test_wait_event() #0 {
entry:
  call void @llvm.amdgcn.s.wait.event.export.ready() #0
  ret void
}

declare void @llvm.amdgcn.s.wait.event.export.ready() #0

attributes #0 = { nounwind }
