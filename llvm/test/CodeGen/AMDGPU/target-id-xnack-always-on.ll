; gfx1250, gfx1251, and gfx12-5-generic have xnack always on because they don't
; support on/off modes (no FeatureXNACKOnOffModes). The target ID should not
; include xnack modifiers regardless of -mattr settings.

; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck --check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa < %s | FileCheck --check-prefix=GFX1251 %s
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa < %s | FileCheck --check-prefix=GFX125GEN %s

; Even with -mattr=+xnack or -mattr=-xnack, the target ID doesn't change
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -mattr=+xnack < %s | FileCheck --check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -mattr=-xnack < %s | FileCheck --check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -mattr=+xnack < %s | FileCheck --check-prefix=GFX1251 %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -mattr=-xnack < %s | FileCheck --check-prefix=GFX1251 %s
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa -mattr=+xnack < %s | FileCheck --check-prefix=GFX125GEN %s
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa -mattr=-xnack < %s | FileCheck --check-prefix=GFX125GEN %s

; GFX1250: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
; GFX1251: .amdgcn_target "amdgpu12.51-amd-amdhsa-unknown-gfx1251"
; GFX125GEN: .amdgcn_target "amdgpu12.5-amd-amdhsa-unknown-gfx12-5-generic"

define void @func0() {
entry:
  ret void
}
