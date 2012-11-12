; RUN: llc < %s -tailcallopt -mtriple=x86_64-linux-gnu | FileCheck %s

; Check the GHC call convention works (x86-64)

@base  = external global i64 ; assigned to register: R13
@sp    = external global i64 ; assigned to register: RBP
@hp    = external global i64 ; assigned to register: R12
@r1    = external global i64 ; assigned to register: RBX
@r2    = external global i64 ; assigned to register: R14
@r3    = external global i64 ; assigned to register: RSI
@r4    = external global i64 ; assigned to register: RDI
@r5    = external global i64 ; assigned to register: R8
@r6    = external global i64 ; assigned to register: R9
@splim = external global i64 ; assigned to register: R15

@f1 = external global float  ; assigned to register: XMM1
@f2 = external global float  ; assigned to register: XMM2
@f3 = external global float  ; assigned to register: XMM3
@f4 = external global float  ; assigned to register: XMM4
@d1 = external global double ; assigned to register: XMM5
@d2 = external global double ; assigned to register: XMM6

define void @zap(i64 %a, i64 %b) nounwind {
entry:
  ; CHECK:      movq %rdi, %r13
  ; CHECK-NEXT: movq %rsi, %rbp
  ; CHECK-NEXT: callq addtwo
  %0 = call cc 10 i64 @addtwo(i64 %a, i64 %b)
  ; CHECK:      callq foo
  call void @foo() nounwind
  ret void
}

define cc 10 i64 @addtwo(i64 %x, i64 %y) nounwind {
entry:
  ; CHECK:      leaq (%r13,%rbp), %rax
  %0 = add i64 %x, %y
  ; CHECK-NEXT: ret
  ret i64 %0
}

define cc 10 void @foo() nounwind {
entry:
  ; CHECK: movq base(%rip), %r13
  ; CHECK-NEXT: movq sp(%rip), %rbp
  ; CHECK-NEXT: movq hp(%rip), %r12
  ; CHECK-NEXT: movq r1(%rip), %rbx
  ; CHECK-NEXT: movq r2(%rip), %r14
  ; CHECK-NEXT: movq r3(%rip), %rsi
  ; CHECK-NEXT: movq r4(%rip), %rdi
  ; CHECK-NEXT: movq r5(%rip), %r8
  ; CHECK-NEXT: movq r6(%rip), %r9
  ; CHECK-NEXT: movq splim(%rip), %r15
  ; CHECK-NEXT: movss f1(%rip), %xmm1
  ; CHECK-NEXT: movss f2(%rip), %xmm2
  ; CHECK-NEXT: movss f3(%rip), %xmm3
  ; CHECK-NEXT: movss f4(%rip), %xmm4
  ; CHECK-NEXT: movsd d1(%rip), %xmm5
  ; CHECK-NEXT: movsd d2(%rip), %xmm6
  %0 = load double* @d2
  %1 = load double* @d1
  %2 = load float* @f4
  %3 = load float* @f3
  %4 = load float* @f2
  %5 = load float* @f1
  %6 = load i64* @splim
  %7 = load i64* @r6
  %8 = load i64* @r5
  %9 = load i64* @r4
  %10 = load i64* @r3
  %11 = load i64* @r2
  %12 = load i64* @r1
  %13 = load i64* @hp
  %14 = load i64* @sp
  %15 = load i64* @base
  ; CHECK: jmp bar
  tail call cc 10 void @bar( i64 %15, i64 %14, i64 %13, i64 %12, i64 %11,
                             i64 %10, i64 %9, i64 %8, i64 %7, i64 %6,
                             float %5, float %4, float %3, float %2, double %1,
                             double %0 ) nounwind
  ret void
}

declare cc 10 void @bar(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
                        float, float, float, float, double, double)

