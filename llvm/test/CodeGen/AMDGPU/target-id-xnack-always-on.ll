; gfx1250, gfx1251, and gfx12-5-generic have xnack always on because they don't
; support on/off modes (no FeatureXNACKOnOffModes). The target ID should not
; include xnack modifiers regardless of -mattr settings.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1251 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx12-5-generic < %s | FileCheck --check-prefix=CHECK %s

; Even with --amdgpu-xnack=true or --amdgpu-xnack=false, the target ID doesn't change
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 --amdgpu-xnack=true < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 --amdgpu-xnack=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1251 --amdgpu-xnack=true < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1251 --amdgpu-xnack=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx12-5-generic --amdgpu-xnack=true < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx12-5-generic --amdgpu-xnack=false < %s | FileCheck %s

; CHECK: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx{{1250|1251|12-5-generic}}"

define void @func0() {
entry:
  ret void
}
