; RUN: not llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR -implicit-check-not=error %s

; ERR: error: inline assembly requires more registers than available
; ERR-NOT: ERROR
; ERR-NOT: Bad machine code

; This test requires respecting undef on the spill source operand when
; expanding the pseudos to avoid all verifier errors

%asm.output = type { <16 x i32>, <8 x i32>, <4 x i32>, <3 x i32>, <3 x i32> }

define void @foo(<32 x i32> addrspace(1)* %arg) #0 {
  %agpr0 = call i32 asm sideeffect "; def $0","=${a0}"()
  %asm = call %asm.output asm sideeffect "; def $0 $1 $2 $3 $4","=v,=v,=v,=v,=v"()
  %vgpr0 = extractvalue %asm.output %asm, 0
  %vgpr1 = extractvalue %asm.output %asm, 1
  %vgpr2 = extractvalue %asm.output %asm, 2
  %vgpr3 = extractvalue %asm.output %asm, 3
  %vgpr4 = extractvalue %asm.output %asm, 4
  call void asm sideeffect "; clobber", "~{a[0:31]},~{v[0:31]}"()
  call void asm sideeffect "; use $0","v"(<16 x i32> %vgpr0)
  call void asm sideeffect "; use $0","v"(<8 x i32> %vgpr1)
  call void asm sideeffect "; use $0","v"(<4 x i32> %vgpr2)
  call void asm sideeffect "; use $0","v"(<3 x i32> %vgpr3)
  call void asm sideeffect "; use $0","v"(<3 x i32> %vgpr4)
  call void asm sideeffect "; use $0","{a1}"(i32 %agpr0)
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="8,8" }
