; RUN: llc < %s -tailcallopt -mtriple=x86_64-linux-gnu | FileCheck %s

; Check the GHC call convention works (x86-64)

@base  = external global i64 ; assigned to register: R13
@sp    = external global i64 ; assigned to register: rbp
@hp    = external global i64 ; assigned to register: R12
@r1    = external global i64 ; assigned to register: rbx
@r2    = external global i64 ; assigned to register: R14
@r3    = external global i64 ; assigned to register: rsi
@r4    = external global i64 ; assigned to register: rdi
@r5    = external global i64 ; assigned to register: R8
@r6    = external global i64 ; assigned to register: R9
@splim = external global i64 ; assigned to register: R15

@f1 = external global float  ; assigned to register: xmm1
@f2 = external global float  ; assigned to register: xmm2
@f3 = external global float  ; assigned to register: xmm3
@f4 = external global float  ; assigned to register: xmm4
@d1 = external global double ; assigned to register: xmm5
@d2 = external global double ; assigned to register: xmm6

define void @zap(i64 %a, i64 %b) nounwind {
entry:
  ; CHECK:      movq %rsi, %rbp
  ; CHECK-NEXT: movq %rdi, %r13
  ; CHECK-NEXT: callq addtwo
  %0 = call ghccc i64 @addtwo(i64 %a, i64 %b)
  ; CHECK:      callq foo
  call void @foo() nounwind
  ret void
}

define ghccc i64 @addtwo(i64 %x, i64 %y) nounwind {
entry:
  ; CHECK:      leaq (%r13,%rbp), %rax
  %0 = add i64 %x, %y
  ; CHECK-NEXT: ret
  ret i64 %0
}

define ghccc void @foo() nounwind {
entry:
  ; CHECK:      movsd d2(%rip), %xmm6
  ; CHECK-NEXT: movsd d1(%rip), %xmm5
  ; CHECK-NEXT: movss f4(%rip), %xmm4
  ; CHECK-NEXT: movss f3(%rip), %xmm3
  ; CHECK-NEXT: movss f2(%rip), %xmm2
  ; CHECK-NEXT: movss f1(%rip), %xmm1
  ; CHECK-NEXT: movq splim(%rip), %r15
  ; CHECK-NEXT: movq r6(%rip), %r9
  ; CHECK-NEXT: movq r5(%rip), %r8
  ; CHECK-NEXT: movq r4(%rip), %rdi
  ; CHECK-NEXT: movq r3(%rip), %rsi
  ; CHECK-NEXT: movq r2(%rip), %r14
  ; CHECK-NEXT: movq r1(%rip), %rbx
  ; CHECK-NEXT: movq hp(%rip), %r12
  ; CHECK-NEXT: movq sp(%rip), %rbp
  ; CHECK-NEXT: movq base(%rip), %r13
  %0 = load double, double* @d2
  %1 = load double, double* @d1
  %2 = load float, float* @f4
  %3 = load float, float* @f3
  %4 = load float, float* @f2
  %5 = load float, float* @f1
  %6 = load i64, i64* @splim
  %7 = load i64, i64* @r6
  %8 = load i64, i64* @r5
  %9 = load i64, i64* @r4
  %10 = load i64, i64* @r3
  %11 = load i64, i64* @r2
  %12 = load i64, i64* @r1
  %13 = load i64, i64* @hp
  %14 = load i64, i64* @sp
  %15 = load i64, i64* @base
  ; CHECK: jmp bar
  tail call ghccc void @bar( i64 %15, i64 %14, i64 %13, i64 %12, i64 %11,
                             i64 %10, i64 %9, i64 %8, i64 %7, i64 %6,
                             float %5, float %4, float %3, float %2, double %1,
                             double %0 ) nounwind
  ret void
}

declare ghccc void @bar(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
                        float, float, float, float, double, double)
