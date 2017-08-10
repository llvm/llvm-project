; RUN: llc < %s -stackrealign -stack-alignment=32 -mattr=+avx -mtriple=x86_64-apple-darwin10 | FileCheck %s
; PR11468

define void @f(i64 %sz) uwtable {
entry:
  %a = alloca i32, align 32
  store volatile i32 0, i32* %a, align 32
  ; force to push r14 on stack
  call void asm sideeffect "nop", "~{r14},~{dirflag},~{fpsr},~{flags}"() nounwind, !srcloc !0
  ret void

; CHECK: _f
; CHECK: pushq %rbp
; CHECK: .cfi_offset %rbp, -16
; CHECK: movq %rsp, %rbp
; CHECK: .cfi_def_cfa_register %rbp

; We first push register on stack, and then realign it, so that
; .cfi_offset value is correct
; CHECK: pushq %r14
; CHECK: andq $-32, %rsp
; CHECK: .cfi_offset %r14, -24

; Restore %rsp from %rbp and subtract the total size of saved regsiters.
; CHECK: leaq -8(%rbp), %rsp

; Pop saved registers.
; CHECK: popq %r14
; CHECK: popq %rbp
}

!0 = !{i32 125}

