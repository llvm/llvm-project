; RUN: llc < %s -stackrealign -mtriple i386-apple-darwin -mcpu=i486 | FileCheck %s

%struct.foo = type { [88 x i8] }

declare void @bar(i8* nocapture, %struct.foo* align 4 byval(%struct.foo)) nounwind

; PR19012
; Don't clobber %esi if we have inline asm that clobbers %esp.
define void @test1(%struct.foo* nocapture %x, i32 %y, i8* %z) nounwind {
  call void @bar(i8* %z, %struct.foo* align 4 byval(%struct.foo) %x)
  call void asm sideeffect inteldialect "xor esp, esp", "=*m,~{flags},~{esp},~{esp},~{dirflag},~{fpsr},~{flags}"(i8* %z)
  ret void

; CHECK-LABEL: test1:
; CHECK: movl %esp, %esi
; CHECK-NOT: rep;movsl
}
