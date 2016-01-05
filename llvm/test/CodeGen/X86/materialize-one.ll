; RUN: llc -mtriple=i686-unknown-linux-gnu -mattr=+cmov %s -o - | FileCheck %s --check-prefix=CHECK32
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+cmov %s -o - | FileCheck %s --check-prefix=CHECK64

define i32 @one32() optsize {
entry:
  ret i32 1

; CHECK32-LABEL: one32
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  ret

; FIXME: Figure out the best approach in 64-bit mode.
; CHECK64-LABEL: one32
; CHECK64:       movl $1, %eax
; CHECK64-NEXT:  retq
}

define i32 @minus_one32() optsize {
entry:
  ret i32 -1

; CHECK32-LABEL: minus_one32
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  ret
}

define i16 @one16() optsize {
entry:
  ret i16 1

; CHECK32-LABEL: one16
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  retl
}

define i16 @minus_one16() optsize {
entry:
  ret i16 -1

; CHECK32-LABEL: minus_one16
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  retl
}

define i32 @test_rematerialization() optsize {
entry:
  ; Materialize -1 (thiscall forces it into %ecx).
  tail call x86_thiscallcc void @f(i32 -1)

  ; Clobber all registers except %esp, leaving nowhere to store the -1 besides
  ; spilling it to the stack.
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{edi},~{esi},~{ebp},~{dirflag},~{fpsr},~{flags}"()

  ; -1 should be re-materialized here instead of getting spilled above.
  ret i32 -1

; CHECK32-LABEL: test_rematerialization
; CHECK32:       xorl %ecx, %ecx
; CHECK32-NEXT:  decl %ecx
; CHECK32:       calll
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NOT:   %eax
; CHECK32:       retl
}

define i32 @test_rematerialization2(i32 %x) optsize {
entry:
  ; Materialize -1 (thiscall forces it into %ecx).
  tail call x86_thiscallcc void @f(i32 -1)

  ; Clobber all registers except %esp, leaving nowhere to store the -1 besides
  ; spilling it to the stack.
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{edi},~{esi},~{ebp},~{dirflag},~{fpsr},~{flags}"()

  ; Define eflags.
  %a = icmp ne i32 %x, 123
  %b = zext i1 %a to i32
  ; Cause -1 to be rematerialized right in front of the cmov, which needs eflags.
  ; It must therefore not use the xor-dec lowering.
  %c = select i1 %a, i32 %b, i32 -1
  ret i32 %c

; CHECK32-LABEL: test_rematerialization2
; CHECK32:       xorl %ecx, %ecx
; CHECK32-NEXT:  decl %ecx
; CHECK32:       calll
; CHECK32:       cmpl
; CHECK32:       setne
; CHECK32-NOT:   xorl
; CHECK32:       movl $-1
; CHECK32:       cmov
; CHECK32:       retl
}

declare x86_thiscallcc void @f(i32)
