; RUN: llc -march=x86 -mcpu=i686 -mattr=+mmx < %s | FileCheck %s
; <rdar://problem/10106006>

define void @func() nounwind ssp {
; CHECK:  psrlw %mm0, %mm1
entry:
  call void asm sideeffect "psrlw $0, %mm1", "y,~{dirflag},~{fpsr},~{flags}"(i32 8) nounwind
  unreachable

bb367:                                            ; preds = %entry                                                                                                                 
  ret void
}
