; RUN: llc -mtriple=i686-unknown-linux-gnu -enable-misched=false < %s | FileCheck %s

define fastcc i8 @fold32to8(i32 %add, i8 %spill) {
; CHECK-LABEL: fold32to8:
; CHECK:    movl %ecx, (%esp) # 4-byte Spill
; CHECK:    subb (%esp), %dl  # 1-byte Folded Reload
entry:
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edi},~{esi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %trunc = trunc i32 %add to i8
  %sub = sub i8 %spill, %trunc
  ret i8 %sub
}

; Do not fold a 1-byte store into a 4-byte spill slot
define fastcc i8 @nofold(i32 %add, i8 %spill) {
; CHECK-LABEL: nofold:
; CHECK:    movl %edx, (%esp) # 4-byte Spill
; CHECK:    movl (%esp), %eax # 4-byte Reload
; CHECK:    subb %cl, %al
entry:
  tail call void asm sideeffect "", "~{eax},~{ebx},~{edx},~{edi},~{esi},~{ebp},~{dirflag},~{fpsr},~{flags}"()
  %trunc = trunc i32 %add to i8
  %sub = sub i8 %spill, %trunc
  ret i8 %sub
}
