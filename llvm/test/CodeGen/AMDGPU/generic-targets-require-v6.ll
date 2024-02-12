; RUN: not llc -march=amdgcn -mcpu=gfx9-generic --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX9-V5 %s
; RUN: not llc -march=amdgcn -mcpu=gfx10.1-generic --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX101-V5 %s
; RUN: not llc -march=amdgcn -mcpu=gfx10.3-generic --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX103-V5 %s
; RUN: not llc -march=amdgcn -mcpu=gfx11-generic --amdhsa-code-object-version=5 -o - %s 2>&1 | FileCheck --check-prefix=GFX11-V5 %s

; RUN: llc -march=amdgcn -mcpu=gfx9-generic --amdhsa-code-object-version=6 -o - %s
; RUN: llc -march=amdgcn -mcpu=gfx10.1-generic --amdhsa-code-object-version=6 -o - %s
; RUN: llc -march=amdgcn -mcpu=gfx10.3-generic --amdhsa-code-object-version=6 -o - %s
; RUN: llc -march=amdgcn -mcpu=gfx11-generic --amdhsa-code-object-version=6 -o - %s

; GFX9-V5:   gfx9-generic is only available on code object version 6 or better
; GFX101-V5: gfx10.1-generic is only available on code object version 6 or better
; GFX103-V5: gfx10.3-generic is only available on code object version 6 or better
; GFX11-V5:  gfx11-generic is only available on code object version 6 or better

define void @foo() {
  ret void
}
