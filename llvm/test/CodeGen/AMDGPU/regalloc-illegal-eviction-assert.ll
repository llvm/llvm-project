; RUN: not llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs -o - %s 2>%t.err | FileCheck -implicit-check-not=error %s
; RUN: FileCheck -check-prefix=ERR %s < %t.err

; This testcase would fail on an "illegal eviction". If the assert was
; relaxed to allow equivalent cascade numbers, it would infinite loop.

; ERR: error: inline assembly requires more registers than available

%asm.output = type { <16 x i32>, <8 x i32>, <5 x i32>, <4 x i32>, <16 x i32> }

; CHECK-LABEL: {{^}}illegal_eviction_assert:
; CHECK: ; def v[4:19] v[20:27] v[0:4] v[0:3] a[0:15]
; CHECK: ; clobber
; CHECK: ; use v[4:19] v[20:27] v[0:4] v[0:3] a[1:16]
define void @illegal_eviction_assert(ptr addrspace(1) %arg) #0 {
  ;%agpr0 = call i32 asm sideeffect "; def $0","=${a0}"()
  %asm = call %asm.output asm sideeffect "; def $0 $1 $2 $3 $4","=v,=v,=v,=v,={a[0:15]}"()
  %vgpr0 = extractvalue %asm.output %asm, 0
  %vgpr1 = extractvalue %asm.output %asm, 1
  %vgpr2 = extractvalue %asm.output %asm, 2
  %vgpr3 = extractvalue %asm.output %asm, 3
  %agpr0 = extractvalue %asm.output %asm, 4
  call void asm sideeffect "; clobber", "~{v[0:31]}"()
  call void asm sideeffect "; use $0 $1 $2 $3 $4","v,v,v,v,{a[1:16]}"(<16 x i32> %vgpr0, <8 x i32> %vgpr1, <5 x i32> %vgpr2, <4 x i32> %vgpr3, <16 x i32> %agpr0)
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="8,8" }
