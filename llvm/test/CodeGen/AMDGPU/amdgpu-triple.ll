; RUN: llc -mtriple=amdgpu6-amd-amdhsa < %s | FileCheck -check-prefixes=GFX6 %s
; RUN: llc -mtriple=amdgpu6-amd-amdhsa -mcpu=gfx601 < %s | FileCheck -check-prefixes=GFX601 %s

; RUN: llc -mtriple=amdgpu7-amd-amdhsa < %s | FileCheck -check-prefixes=GFX7 %s
; RUN: llc -mtriple=amdgpu7-amd-amdhsa -mcpu=gfx701 < %s | FileCheck -check-prefixes=GFX701 %s

; RUN: llc -mtriple=amdgpu8-amd-amdhsa < %s | FileCheck -check-prefixes=GFX8 %s
; RUN: llc -mtriple=amdgpu8-amd-amdhsa -mcpu=gfx803 < %s | FileCheck -check-prefixes=GFX803 %s

; RUN: llc -mtriple=amdgpu9-amd-amdhsa < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: llc -mtriple=amdgpu9.08-amd-amdhsa -mcpu=gfx908 < %s | FileCheck -check-prefixes=GFX908 %s

; RUN: llc -mtriple=amdgpu9.4-amd-amdhsa < %s | FileCheck -check-prefixes=GFX94 %s
; RUN: llc -mtriple=amdgpu9.4-amd-amdhsa -mcpu=gfx942 < %s | FileCheck -check-prefixes=GFX942 %s

; RUN: llc -mtriple=amdgpu10-amd-amdhsa < %s | FileCheck -check-prefixes=GFX10 %s
; RUN: llc -mtriple=amdgpu10.1-amd-amdhsa < %s | FileCheck -check-prefixes=GFX10_1 %s

; RUN: llc -mtriple=amdgpu10.1-amd-amdhsa -mcpu=gfx1012 < %s | FileCheck -check-prefixes=GFX1012 %s

; RUN: llc -mtriple=amdgpu10.3-amd-amdhsa < %s | FileCheck -check-prefixes=GFX10_3 %s
; RUN: llc -mtriple=amdgpu10.3-amd-amdhsa -mcpu=gfx1031 < %s | FileCheck -check-prefixes=GFX1031 %s

; RUN: llc -mtriple=amdgpu11-amd-amdhsa < %s | FileCheck -check-prefixes=GFX11 %s
; RUN: llc -mtriple=amdgpu11-amd-amdhsa -mcpu=gfx1101 < %s | FileCheck -check-prefixes=GFX1101 %s

; RUN: llc -mtriple=amdgpu11.7-amd-amdhsa < %s | FileCheck -check-prefixes=GFX117 %s
; RUN: llc -mtriple=amdgpu11.7-amd-amdhsa -mcpu=gfx1170 < %s | FileCheck -check-prefixes=GFX1170 %s

; RUN: llc -mtriple=amdgpu12-amd-amdhsa < %s | FileCheck -check-prefixes=GFX12 %s
; RUN: llc -mtriple=amdgpu12-amd-amdhsa -mcpu=gfx1200 < %s | FileCheck -check-prefixes=GFX1200 %s

; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa < %s | FileCheck -check-prefixes=GFX125 %s
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa -mcpu=gfx1250 < %s | FileCheck -check-prefixes=GFX1250 %s

; RUN: llc -mtriple=amdgpu13-amd-amdhsa < %s | FileCheck -check-prefixes=GFX13 %s
; RUN: llc -mtriple=amdgpu13-amd-amdhsa -mcpu=gfx1310 < %s | FileCheck -check-prefixes=GFX1310 %s

; GFX6: .amdgcn_target "amdgpu6-amd-amdhsa-unknown-gfx600"
; GFX601: .amdgcn_target "amdgpu6-amd-amdhsa-unknown-gfx601"

; GFX7: .amdgcn_target "amdgpu7-amd-amdhsa-unknown-gfx700"
; GFX701: .amdgcn_target "amdgpu7-amd-amdhsa-unknown-gfx701"

; GFX8: .amdgcn_target "amdgpu8-amd-amdhsa-unknown-gfx801"
; GFX803: .amdgcn_target "amdgpu8-amd-amdhsa-unknown-gfx803"

; GFX9: .amdgcn_target "amdgpu9-amd-amdhsa-unknown-gfx9-generic"
; GFX908: .amdgcn_target "amdgpu9.08-amd-amdhsa-unknown-gfx908"

; GFX94: .amdgcn_target "amdgpu9.4-amd-amdhsa-unknown-gfx9-4-generic"
; GFX942: .amdgcn_target "amdgpu9.4-amd-amdhsa-unknown-gfx942"

; FIXME: Normalize this
; GFX10: .amdgcn_target "amdgpu10-amd-amdhsa-unknown-gfx10-1-generic"
; GFX10_1: .amdgcn_target "amdgpu10.1-amd-amdhsa-unknown-gfx10-1-generic"
; GFX1012: .amdgcn_target "amdgpu10.1-amd-amdhsa-unknown-gfx1012"

; GFX10_3: .amdgcn_target "amdgpu10.3-amd-amdhsa-unknown-gfx10-3-generic"
; GFX1031: .amdgcn_target "amdgpu10.3-amd-amdhsa-unknown-gfx1031"

; GFX11: .amdgcn_target "amdgpu11-amd-amdhsa-unknown-gfx11-generic"
; GFX1101: .amdgcn_target "amdgpu11-amd-amdhsa-unknown-gfx1101"

; GFX117: .amdgcn_target "amdgpu11.7-amd-amdhsa-unknown-gfx11-7-generic"
; GFX1170: .amdgcn_target "amdgpu11.7-amd-amdhsa-unknown-gfx1170"

; GFX12: .amdgcn_target "amdgpu12-amd-amdhsa-unknown-gfx12-generic"
; GFX1200: .amdgcn_target "amdgpu12-amd-amdhsa-unknown-gfx1200"

; GFX125: .amdgcn_target "amdgpu12.5-amd-amdhsa-unknown-gfx12-5-generic"
; GFX1250: .amdgcn_target "amdgpu12.5-amd-amdhsa-unknown-gfx1250"

; GFX13: .amdgcn_target "amdgpu13-amd-amdhsa-unknown-gfx13-generic"
; GFX1310: .amdgcn_target "amdgpu13-amd-amdhsa-unknown-gfx1310"
define amdgpu_kernel void @kernel() {
  ret void
}
