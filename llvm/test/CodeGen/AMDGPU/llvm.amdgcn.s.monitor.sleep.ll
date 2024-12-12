; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

declare void @llvm.amdgcn.s.monitor.sleep(i16)

; GCN-LABEL: {{^}}test_monitor_sleep_1:
; GCN: s_monitor_sleep 1
define amdgpu_ps void @test_monitor_sleep_1() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 1)
  ret void
}

; FIXME: 0x8000 would look better

; GCN-LABEL: {{^}}test_monitor_sleep_forever:
; GCN: s_monitor_sleep 0xffff8000
define amdgpu_ps void @test_monitor_sleep_forever() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 32768)
  ret void
}
