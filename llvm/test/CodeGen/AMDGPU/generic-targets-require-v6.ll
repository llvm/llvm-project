; RUN: not llc -mtriple=amdgpu9 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX9-V5 %s
; RUN: not llc -mtriple=amdgpu9.4 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX9-4-V5 %s
; RUN: not llc -mtriple=amdgpu10.1 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX101-V5 %s
; RUN: not llc -mtriple=amdgpu10.3 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX103-V5 %s
; RUN: not llc -mtriple=amdgpu11 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX11-V5 %s
; RUN: not llc -mtriple=amdgpu11.7 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX117-V5 %s
; RUN: not llc -mtriple=amdgpu12 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX12-V5 %s
; RUN: not llc -mtriple=amdgpu12.5 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX125-V5 %s
; RUN: not llc -mtriple=amdgpu13 --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX13-V5 %s

; RUN: llc -mtriple=amdgpu9 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu9.4 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu10.1 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu10.3 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu11 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu11.7 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu12 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu12.5 --amdhsa-code-object-version=6 -o - %s
; RUN: llc -mtriple=amdgpu13 --amdhsa-code-object-version=6 -o - %s

; GFX9-V5:   gfx9-generic is only available on code object version 6 or better
; GFX9-4-V5: gfx9-4-generic is only available on code object version 6 or better
; GFX101-V5: gfx10-1-generic is only available on code object version 6 or better
; GFX103-V5: gfx10-3-generic is only available on code object version 6 or better
; GFX11-V5:  gfx11-generic is only available on code object version 6 or better
; GFX117-V5: gfx11-7-generic is only available on code object version 6 or better
; GFX12-V5:  gfx12-generic is only available on code object version 6 or better
; GFX125-V5: gfx12-5-generic is only available on code object version 6 or better
; GFX13-V5:  gfx13-generic is only available on code object version 6 or better

define void @foo() {
  ret void
}
