; NOTE: Do not autogenerate. Check the diagnostic from normal instruction emission.
; The target verifier must reject direct IR even when MachineVerifier is disabled.
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.50 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -global-isel-abort=1 -verify-machineinstrs=0 -mtriple=amdgpu12.50 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.50s < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -global-isel-abort=1 -verify-machineinstrs=0 -mtriple=amdgpu12.50s < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.51 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -global-isel-abort=1 -verify-machineinstrs=0 -mtriple=amdgpu12.51 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.5 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -global-isel-abort=1 -verify-machineinstrs=0 -mtriple=amdgpu12.5 < %s 2>&1 | FileCheck %s

; CHECK: error: Illegal instruction detected: sleep duration must be in range [0, 127] on this target
; CHECK: error: Illegal instruction detected: sleep duration must be in range [0, 127] on this target

declare void @llvm.amdgcn.s.monitor.sleep(i16)

define amdgpu_ps void @sleep_128() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 128)
  ret void
}

define amdgpu_ps void @sleep_200() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 200)
  ret void
}
