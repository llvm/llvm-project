; RUN: llc < %s -stackrealign -mtriple i386-apple-darwin -mcpu=i486 | FileCheck %s

%struct.foo = type { [88 x i8] }

declare void @bar(ptr nocapture, ptr align 4 byval(%struct.foo)) nounwind

; PR19012
; Don't clobber %esi if we have inline asm that clobbers %esp.
define void @test1(ptr nocapture %x, i32 %y, ptr %z) nounwind {
  call void @bar(ptr %z, ptr align 4 byval(%struct.foo) %x)
  call void asm sideeffect inteldialect "xor esp, esp", "=*m,~{flags},~{esp},~{esp},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i8) %z)
  ret void

; CHECK-LABEL: test1:
; CHECK: movl %esp, %esi
; CHECK-NOT: rep;movsl
}
