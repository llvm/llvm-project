;RUN: llc -O2 -mtriple=thumbv7a-linux-gnueabihf -arm-implicit-it=always %s -o - | FileCheck %s
;RUN: llc -O2 -mtriple=thumbv7a-linux-gnueabihf -no-integrated-as %s -o - | FileCheck %s

; Check that we do not produce a CBZ instruction to jump over the inline
; assembly as the distance is too far if the implicit IT instructions are
; added.

define void @f0(i32 %p1, i32 %p2, i32 %p3) nounwind {
entry:
  %cmp = icmp eq i32 %p1, 0
  br i1 %cmp, label %if.else, label %if.then

if.then:
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  tail call void asm sideeffect "movseq r0, #0\0A", ""()
  br label %if.end

if.else:
  tail call void asm sideeffect "nop\0A", ""()
  br label %if.end

if.end:
  ret void
}
; CHECK-LABEL: f0:
; CHECK: beq .LBB0_{{[0-9]+}}

