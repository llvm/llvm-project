; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -passes=amdgpu-lower-intrinsics -S %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -O2 %s -o - | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -O2 %s -o - | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx11-generic -O2 %s -o - | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1170 -O2 %s -o - | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -mattr=-cdbg-sys-or-user-branch -O2 %s -o - | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -O2 -global-isel -global-isel-abort=1 %s -o - | FileCheck %s --check-prefix=GCN

declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @unsupported_subtarget() {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %enabled, label %debug, label %normal

debug:
  br label %normal

normal:
  ret void
}

; IR-LABEL: define amdgpu_kernel void @unsupported_subtarget()
; IR: entry:
; IR-NEXT: br i1 false, label %debug, label %normal
; IR-NOT: call i1 @llvm.is.debugging.enabled()
; IR: ret void

; GCN-LABEL: unsupported_subtarget:
; GCN-NOT: s_cbranch_cdbg
; GCN-NOT: s_getreg
; GCN: s_endpgm
