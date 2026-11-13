; NOTE: Do not autogenerate. Check diagnostics rather than assembly.
; Disable MachineVerifier to exercise the check during code emission.
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.50 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu12.50 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.50s < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu12.50s < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.51 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu12.51 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu12.5 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu12.5 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu13 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu13 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -verify-machineinstrs=0 -mtriple=amdgpu13.10 < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -verify-machineinstrs=0 -mtriple=amdgpu13.10 < %s 2>&1 | FileCheck %s

; CHECK: error: Illegal instruction detected: sleep duration must be in range [0, 127] on this target
; CHECK: error: Illegal instruction detected: sleep duration must be in range [0, 127] on this target

define amdgpu_ps void @sleep_128() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 128)
  ret void
}

define amdgpu_ps void @sleep_200() {
  call void @llvm.amdgcn.s.monitor.sleep(i16 200)
  ret void
}
