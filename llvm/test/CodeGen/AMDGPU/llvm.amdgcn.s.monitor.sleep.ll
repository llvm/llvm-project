; NOTE: Do not autogenerate. Full function checks obscure the target-dependent immediate.
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50 < %s | FileCheck --check-prefixes=GCN,GFX1250 %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.50 < %s | FileCheck --check-prefixes=GCN,GFX1250 %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.50s < %s | FileCheck --check-prefixes=GCN,GFX1250 %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.50s < %s | FileCheck --check-prefixes=GCN,GFX1250 %s
; RUN: llc -global-isel=0 -mtriple=amdgpu12.51 < %s | FileCheck --check-prefixes=GCN,GFX1251 %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.51 < %s | FileCheck --check-prefixes=GCN,GFX1251 %s

; RUN: llc -global-isel=0 -mtriple=amdgpu12.50 -mattr=-no-sleep-forever < %s | FileCheck --check-prefixes=GCN,GFX1251 %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.50 -mattr=-no-sleep-forever < %s | FileCheck --check-prefixes=GCN,GFX1251 %s

; RUN: llc -global-isel=0 -mtriple=amdgpu13 < %s | FileCheck --check-prefixes=GCN,GFX1251 %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu13 < %s | FileCheck --check-prefixes=GCN,GFX1251 %s

; RUN: llc -global-isel=0 -mtriple=amdgpu12.5 < %s | FileCheck --check-prefixes=GCN,GFX1251 %s
; RUN: llc -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu12.5 < %s | FileCheck --check-prefixes=GCN,GFX1251 %s

declare void @llvm.amdgcn.s.monitor.sleep(i16)

; GCN-LABEL: {{^}}test_monitor_sleep_1:
; GCN: s_monitor_sleep 1
define amdgpu_ps void @test_monitor_sleep_1() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 1)
  ret void
}

; GCN-LABEL: {{^}}test_monitor_sleep_127:
; GCN: s_monitor_sleep 0x7f
define amdgpu_ps void @test_monitor_sleep_127() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 127)
  ret void
}

; FIXME: 0x8000 would look better

; GCN-LABEL: {{^}}test_monitor_sleep_forever:
; GFX1250: s_monitor_sleep 0x7f
; GFX1251: s_monitor_sleep 0xffff8000
define amdgpu_ps void @test_monitor_sleep_forever() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 32768)
  ret void
}

; GCN-LABEL: {{^}}test_monitor_sleep_forever_max_duration:
; GFX1250: s_monitor_sleep 0x7f
; GFX1251: s_monitor_sleep 0xffff807f
define amdgpu_ps void @test_monitor_sleep_forever_max_duration() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 32895)
  ret void
}
