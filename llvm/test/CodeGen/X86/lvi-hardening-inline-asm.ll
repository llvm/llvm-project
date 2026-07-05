; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown -mattr=+lvi-load-hardening -mattr=+lvi-cfi -x86-experimental-lvi-inline-asm-hardening < %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=X86 < %t.out
; RUN: FileCheck %s --check-prefix=WARN < %t.err

; Test module-level assembly
module asm "pop %rbx"
module asm "ret"
; X86:      popq %rbx
; X86-NEXT: lfence
; X86-NEXT: shlq $0, (%rsp)
; X86-NEXT: lfence
; X86-NEXT: retq

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @test_inline_asm() {
entry:
; X86-LABEL: test_inline_asm:
  call void asm sideeffect "mov 0x3fed(%rip),%rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      movq  16365(%rip), %rax
; X86-NEXT: lfence
  call void asm sideeffect "movdqa 0x0(%rip),%xmm0", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      movdqa  (%rip), %xmm0
; X86-NEXT: lfence
  call void asm sideeffect "movslq 0x3e5d(%rip),%rbx", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      movslq  15965(%rip), %rbx
; X86-NEXT: lfence
  call void asm sideeffect "mov (%r12,%rax,8),%rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      movq  (%r12,%rax,8), %rax
; X86-NEXT: lfence
  call void asm sideeffect "movq (24)(%rsi), %r11", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      movq  24(%rsi), %r11
; X86-NEXT: lfence
  call void asm sideeffect "cmove %r12,%rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      cmoveq  %r12, %rax
; X86-NOT:  lfence
  call void asm sideeffect "cmove (%r12),%rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      cmoveq  (%r12), %rax
; X86-NEXT: lfence
  call void asm sideeffect "pop %rbx", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      popq  %rbx
; X86-NEXT: lfence
  call void asm sideeffect "popq %rbx", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      popq  %rbx
; X86-NEXT: lfence
  call void asm sideeffect "xchg (%r12),%rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      xchgq %rax, (%r12)
; X86-NEXT: lfence
  call void asm sideeffect "cmpxchg %r12,(%rax)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      cmpxchgq  %r12, (%rax)
; X86-NEXT: lfence
  call void asm sideeffect "vpxor (%rcx,%rdx,1),%ymm1,%ymm0", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      vpxor (%rcx,%rdx), %ymm1, %ymm0
; X86-NEXT: lfence
  call void asm sideeffect "vpmuludq 0x20(%rsi),%ymm0,%ymm12", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      vpmuludq  32(%rsi), %ymm0, %ymm12
; X86-NEXT: lfence
  call void asm sideeffect "vpexpandq 0x40(%rdi),%zmm8{%k2}{z}", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      vpexpandq 64(%rdi), %zmm8 {%k2} {z}
; X86-NEXT: lfence
  call void asm sideeffect "addq (%r12),%rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      addq  (%r12), %rax
; X86-NEXT: lfence
  call void asm sideeffect "subq Lpoly+0(%rip), %rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      subq  Lpoly+0(%rip), %rax
; X86-NEXT: lfence
  call void asm sideeffect "adcq %r12,(%rax)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      adcq  %r12, (%rax)
; X86-NEXT: lfence
  call void asm sideeffect "negq (%rax)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      negq  (%rax)
; X86-NEXT: lfence
  call void asm sideeffect "incq %rax", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      incq  %rax
; X86-NOT:  lfence
  call void asm sideeffect "mulq (%rax)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      mulq  (%rax)
; X86-NEXT: lfence
  call void asm sideeffect "imulq (%rax),%rdx", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      imulq (%rax), %rdx
; X86-NEXT: lfence
  call void asm sideeffect "shlq $$1,(%rax)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      shlq  (%rax)
; X86-NEXT: lfence
  call void asm sideeffect "shrq $$1,(%rax)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      shrq  (%rax)
; X86-NEXT: lfence
  call void asm sideeffect "repz cmpsb %es:(%rdi),%ds:(%rsi)", "~{dirflag},~{fpsr},~{flags}"() #1
; WARN:      warning: Instruction may be vulnerable to LVI
; WARN-NEXT: repz cmpsb %es:(%rdi),%ds:(%rsi)
; WARN-NEXT: ^
; WARN-NEXT: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
; X86:      rep cmpsb %es:(%rdi), %ds:(%rsi)
; X86-NOT:  lfence
  call void asm sideeffect "repnz scasb", "~{dirflag},~{fpsr},~{flags}"() #1
; WARN:      warning: Instruction may be vulnerable to LVI
; WARN-NEXT: repnz scasb
; WARN-NEXT: ^
; WARN-NEXT: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
; X86:      repne scasb %es:(%rdi), %al
; X86-NOT:  lfence
  call void asm sideeffect "repnz", ""() #1
; WARN:      warning: Instruction may be vulnerable to LVI
; WARN-NEXT: repnz
; WARN-NEXT: ^
; WARN-NEXT: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
  call void asm sideeffect "pinsrw $$0x6,(%eax),%xmm0", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      pinsrw  $6, (%eax), %xmm0
; X86-NEXT: lfence
  call void asm sideeffect "ret", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      shlq $0, (%rsp)
; X86-NEXT: lfence
; X86-NEXT: retq
; X86-NOT:  lfence
  call void asm sideeffect "ret $$8", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      shlq $0, (%rsp)
; X86-NEXT: lfence
; X86-NEXT: retq $8
; X86-NOT:  lfence
  call void asm sideeffect "jmpq *(%rdx)", "~{dirflag},~{fpsr},~{flags}"() #1
; WARN:      warning: Instruction may be vulnerable to LVI
; WARN-NEXT: jmpq *(%rdx)
; WARN-NEXT: ^
; WARN-NEXT: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
; X86:      jmpq *(%rdx)
; X86-NOT:  lfence
  call void asm sideeffect "jmpq *0x100(%rdx)", "~{dirflag},~{fpsr},~{flags}"() #1
; WARN:      warning: Instruction may be vulnerable to LVI
; WARN-NEXT: jmpq *0x100(%rdx)
; WARN-NEXT: ^
; WARN-NEXT: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
; X86:      jmpq *256(%rdx)
; X86-NOT:  lfence
  call void asm sideeffect "callq *200(%rdx)", "~{dirflag},~{fpsr},~{flags}"() #1
; WARN:      warning: Instruction may be vulnerable to LVI
; WARN-NEXT: callq *200(%rdx)
; WARN-NEXT: ^
; WARN-NEXT: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
; X86:      callq *200(%rdx)
; X86-NOT:  lfence
  call void asm sideeffect "fldt 0x8(%rbp)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      fldt  8(%rbp)
; X86-NEXT: lfence
  call void asm sideeffect "fld %st(0)", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      fld %st(0)
; X86-NOT:  lfence
; Test assembler macros
  call void asm sideeffect ".macro mplus1 x\0Aincq (\5Cx)\0A.endm\0Amplus1 %rcx", "~{dirflag},~{fpsr},~{flags}"() #1
; X86:      incq (%rcx)
; X86-NEXT: lfence
  ret void
}

attributes #1 = { nounwind }
