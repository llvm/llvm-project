; NOTE: Do not autogenerate. Check subarchitecture gating and ordering before wait insertion.
; RUN: llc -enable-new-pm -mtriple=amdgpu12.50-amd-amdhsa -O0 -print-pipeline-passes=tree < %s 2>&1 | FileCheck %s --check-prefix=NPM
; RUN: llc -enable-new-pm -mtriple=amdgpu12.51-amd-amdhsa -print-pipeline-passes=tree < %s 2>&1 | FileCheck %s --check-prefix=NPM
; RUN: llc -enable-new-pm -mtriple=amdgpu12.50s-amd-amdhsa -print-pipeline-passes=tree < %s 2>&1 | FileCheck %s --check-prefix=NPM
; RUN: llc -enable-new-pm -mtriple=amdgpu12.5-amd-amdhsa -print-pipeline-passes=tree < %s 2>&1 | FileCheck %s --check-prefix=NPM
; RUN: llc -enable-new-pm -mtriple=amdgpu13.10-amd-amdhsa -print-pipeline-passes=tree < %s 2>&1 | FileCheck %s --check-prefix=NPM-OFF --implicit-check-not=si-insert-async-barrier-arrive-workaround
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -O0 -debug-pass=Structure < %s 2>&1 | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -debug-pass=Structure < %s 2>&1 | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa -debug-pass=Structure < %s 2>&1 | FileCheck %s --check-prefix=LEGACY-OFF --implicit-check-not=si-insert-async-barrier-arrive-workaround
; RUN: llc -mtriple=amdgpu13.10-amd-amdhsa -debug-pass=Structure < %s 2>&1 | FileCheck %s --check-prefix=LEGACY-OFF --implicit-check-not=si-insert-async-barrier-arrive-workaround

; NPM: si-insert-async-barrier-arrive-workaround
; NPM-NEXT: si-insert-waitcnts
; NPM-OFF: si-insert-waitcnts
; LEGACY: SI insert async barrier arrival workaround
; LEGACY: SI insert wait instructions
; LEGACY-OFF: SI insert wait instructions

define amdgpu_kernel void @empty() {
  ret void
}
