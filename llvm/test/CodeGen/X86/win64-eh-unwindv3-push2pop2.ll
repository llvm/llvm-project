; RUN: llc -mtriple=x86_64-unknown-windows-msvc -mattr=+push2pop2 -o - %s | FileCheck %s

; Test that push2/pop2 padding cleanup in the epilog emits the correct
; SEH pseudo for unwind v3. The padding PUSH in the prolog gets an
; SEH_PushReg, so the corresponding cleanup in the epilog (an ADD RSP,8
; or POP into a dead register) must also have an SEH pseudo.

; CHECK:        .seh_unwindversion 3

declare void @a() local_unnamed_addr
declare i32 @c(i32) local_unnamed_addr

; Function with 6 callee-saved GPRs (even count) which triggers push2/pop2
; padding (extra push to align stack for push2).
define dso_local i32 @push2pop2_padding(i32 %x) local_unnamed_addr {
entry:
  call void asm sideeffect "", "~{rbx},~{rbp},~{r12},~{r13},~{r14},~{r15}"()
  %call = tail call i32 @c(i32 %x)
  ret i32 %call
}
; CHECK-LABEL:  push2pop2_padding:
; Prolog: padding push + push2 pairs
; CHECK:        .seh_proc push2pop2_padding
; CHECK:        .seh_pushreg %rax
; V3 uses SEH_Push2Regs for PUSH2 (one SEH pseudo per PUSH2 instruction):
; CHECK:        .seh_push2regs %r15, %r14
; CHECK-NEXT:   push2   %r14, %r15
; CHECK:        .seh_push2regs %r13, %r12
; CHECK-NEXT:   push2   %r12, %r13
; CHECK:        .seh_push2regs %rbp, %rbx
; CHECK-NEXT:   push2   %rbx, %rbp
; CHECK:        .seh_endprologue
;
; Epilog: must have SEH pseudos for all pops AND the padding cleanup.
; CHECK:        .seh_startepilogue
; The GPR pop2 instructions get SEH_Push2Regs pseudos with matching registers:
; CHECK:        .seh_push2regs %rbx, %rbp
; CHECK-NEXT:   pop2    %rbp, %rbx
; CHECK:        .seh_push2regs %r12, %r13
; CHECK-NEXT:   pop2    %r13, %r12
; CHECK:        .seh_push2regs %r14, %r15
; CHECK-NEXT:   pop2    %r15, %r14
; The padding cleanup (pop or add rsp,8) must also have SEH_StackAlloc:
; CHECK:        .seh_stackalloc 8
; CHECK:        .seh_endepilogue

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 3}
