; RUN: llc -mtriple=i686-unknown-linux-gnu -mattr=+cmov %s -o - | FileCheck %s --check-prefix=CHECK32
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+cmov %s -o - | FileCheck %s --check-prefix=CHECK64
; RUN: llc -mtriple=x86_64-pc-win32 -mattr=+cmov %s -o - | FileCheck %s --check-prefix=CHECKWIN64

define i32 @one32_nooptsize() {
entry:
  ret i32 1

; When not optimizing for size, use mov.
; CHECK32-LABEL: one32_nooptsize:
; CHECK32:       movl $1, %eax
; CHECK32-NEXT:  retl
; CHECK64-LABEL: one32_nooptsize:
; CHECK64:       movl $1, %eax
; CHECK64-NEXT:  retq
}

define i32 @one32() optsize {
entry:
  ret i32 1

; CHECK32-LABEL: one32:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  retl

; FIXME: Figure out the best approach in 64-bit mode.
; CHECK64-LABEL: one32:
; CHECK64:       movl $1, %eax
; CHECK64-NEXT:  retq
}

define i32 @one32_pgso() !prof !14 {
entry:
  ret i32 1

; CHECK32-LABEL: one32_pgso:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  retl

; FIXME: Figure out the best approach in 64-bit mode.
; CHECK64-LABEL: one32_pgso:
; CHECK64:       movl $1, %eax
; CHECK64-NEXT:  retq
}

define i32 @one32_minsize() minsize {
entry:
  ret i32 1

; On 32-bit, xor-inc is preferred over push-pop.
; CHECK32-LABEL: one32_minsize:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  retl

; On 64-bit we don't do xor-inc yet, so push-pop it is. Note that we have to
; pop into a 64-bit register even when we just need 32 bits.
; CHECK64-LABEL: one32_minsize:
; CHECK64:       pushq $1
; CHECK64:       .cfi_adjust_cfa_offset 8
; CHECK64:       popq %rax
; CHECK64:       .cfi_adjust_cfa_offset -8
; CHECK64-NEXT:  retq

; On Win64 we can't adjust the stack unless there's a frame pointer.
; CHECKWIN64-LABEL: one32_minsize:
; CHECKWIN64:       movl $1, %eax
; CHECKWIN64-NEXT:  retq
}

define i32 @pr26023() minsize {
entry:
  %x = alloca [120 x i8]
  %0 = getelementptr inbounds [120 x i8], [120 x i8]* %x, i64 0, i64 0
  call void asm sideeffect "", "imr,~{memory},~{dirflag},~{fpsr},~{flags}"(i8* %0)
  %arrayidx = getelementptr inbounds [120 x i8], [120 x i8]* %x, i64 0, i64 119
  store volatile i8 -2, i8* %arrayidx
  call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(i32 5)
  %1 = load volatile i8, i8* %arrayidx
  %conv = sext i8 %1 to i32
  ret i32 %conv

; The function writes to the redzone, so push/pop cannot be used.
; CHECK64-LABEL: pr26023:
; CHECK64:       movl $5, %ecx
; CHECK64:       retq

; 32-bit X86 doesn't have a redzone.
; CHECK32-LABEL: pr26023:
; CHECK32:       pushl $5
; CHECK32:       popl %ecx
; CHECK32:       retl
}


define i64 @one64_minsize() minsize {
entry:
  ret i64 1
; On 64-bit we don't do xor-inc yet, so push-pop it is.
; CHECK64-LABEL: one64_minsize:
; CHECK64:       pushq $1
; CHECK64:       .cfi_adjust_cfa_offset 8
; CHECK64:       popq %rax
; CHECK64:       .cfi_adjust_cfa_offset -8
; CHECK64-NEXT:  retq

; On Win64 we can't adjust the stack unless there's a frame pointer.
; CHECKWIN64-LABEL: one64_minsize:
; CHECKWIN64:       movl $1, %eax
; CHECKWIN64-NEXT:  retq
}

define i32 @minus_one32() optsize {
entry:
  ret i32 -1

; CHECK32-LABEL: minus_one32:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  retl
}

define i32 @minus_one32_pgso() !prof !14 {
entry:
  ret i32 -1

; CHECK32-LABEL: minus_one32_pgso:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  retl
}

define i32 @minus_one32_minsize() minsize {
entry:
  ret i32 -1

; xor-dec is preferred over push-pop.
; CHECK32-LABEL: minus_one32_minsize:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  retl
}

define i16 @one16() optsize {
entry:
  ret i16 1

; CHECK32-LABEL: one16:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  # kill
; CHECK32-NEXT:  retl
}

define i16 @minus_one16() optsize {
entry:
  ret i16 -1

; CHECK32-LABEL: minus_one16:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  # kill
; CHECK32-NEXT:  retl
}

define i16 @one16_pgso() !prof !14 {
entry:
  ret i16 1

; CHECK32-LABEL: one16_pgso:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  incl %eax
; CHECK32-NEXT:  # kill
; CHECK32-NEXT:  retl
}

define i16 @minus_one16_pgso() !prof !14 {
entry:
  ret i16 -1

; CHECK32-LABEL: minus_one16_pgso:
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NEXT:  # kill
; CHECK32-NEXT:  retl
}

define i32 @minus_five32() minsize {
entry:
  ret i32 -5

; CHECK32-LABEL: minus_five32:
; CHECK32: pushl $-5
; CHECK32: popl %eax
; CHECK32: retl
}

define i64 @minus_five64() minsize {
entry:
  ret i64 -5

; CHECK64-LABEL: minus_five64:
; CHECK64: pushq $-5
; CHECK64:       .cfi_adjust_cfa_offset 8
; CHECK64: popq %rax
; CHECK64:       .cfi_adjust_cfa_offset -8
; CHECK64: retq
}

define i32 @rematerialize_minus_one() optsize {
entry:
  ; Materialize -1 (thiscall forces it into %ecx).
  tail call x86_thiscallcc void @f(i32 -1)

  ; Clobber all registers except %esp, leaving nowhere to store the -1 besides
  ; spilling it to the stack.
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{edi},~{esi},~{ebp},~{dirflag},~{fpsr},~{flags}"()

  ; -1 should be re-materialized here instead of getting spilled above.
  ret i32 -1

; CHECK32-LABEL: rematerialize_minus_one
; CHECK32:       xorl %ecx, %ecx
; CHECK32-NEXT:  decl %ecx
; CHECK32:       calll
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NOT:   %eax
; CHECK32:       retl
}

define i32 @rematerialize_minus_one_eflags(i32 %x) optsize {
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

; CHECK32-LABEL: rematerialize_minus_one_eflags
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

define i32 @rematerialize_minus_one_pgso() !prof !14 {
entry:
  ; Materialize -1 (thiscall forces it into %ecx).
  tail call x86_thiscallcc void @f(i32 -1)

  ; Clobber all registers except %esp, leaving nowhere to store the -1 besides
  ; spilling it to the stack.
  tail call void asm sideeffect "", "~{eax},~{ebx},~{ecx},~{edx},~{edi},~{esi},~{ebp},~{dirflag},~{fpsr},~{flags}"()

  ; -1 should be re-materialized here instead of getting spilled above.
  ret i32 -1

; CHECK32-LABEL: rematerialize_minus_one_pgso
; CHECK32:       xorl %ecx, %ecx
; CHECK32-NEXT:  decl %ecx
; CHECK32:       calll
; CHECK32:       xorl %eax, %eax
; CHECK32-NEXT:  decl %eax
; CHECK32-NOT:   %eax
; CHECK32:       retl
}

define i32 @rematerialize_minus_one_eflags_pgso(i32 %x) !prof !14 {
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

; CHECK32-LABEL: rematerialize_minus_one_eflags_pgso
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

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
