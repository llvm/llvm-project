; NOTE: Do not autogenerate. Check diagnostics for reserved duration bits with sleep forever enabled.
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.51 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu12.51 < %s 2>&1 | FileCheck %s

; CHECK: error: Illegal instruction detected: sleep duration must be in range [0, 127] on this target
; CHECK: error: Illegal instruction detected: sleep duration must be in range [0, 127] on this target

define amdgpu_ps void @sleep_forever_128() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 -32640)
  ret void
}

define amdgpu_ps void @sleep_forever_reserved() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 -1)
  ret void
}
