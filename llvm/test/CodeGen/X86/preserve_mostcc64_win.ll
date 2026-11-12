; RUN: sed -e "s/RETTYPE/void/;s/RETVAL//" %s | llc -mtriple=x86_64-win32 -mcpu=corei7 | FileCheck --check-prefixes=ALL,VOID %s
; RUN: sed -e "s/RETTYPE/i32/;s/RETVAL/undef/" %s | llc -mtriple=x86_64-win32 -mcpu=corei7 | FileCheck --check-prefixes=ALL,INT %s
; RUN: sed -e "s/RETTYPE/\{i64\,i64\}/;s/RETVAL/undef/" %s | llc -mtriple=x86_64-win32 -mcpu=corei7 | FileCheck --check-prefixes=ALL,INT128 %s

; Every GPR should be saved, except r11 and return registers.
; XMM registers 6-15 should also be saved.
define preserve_mostcc RETTYPE @preserve_mostcc1(i64, i64, double, double) nounwind {
entry:
;ALL-LABEL:   preserve_mostcc1
;ALL:         pushq %r10
;ALL-NEXT:    pushq %r9
;ALL-NEXT:    pushq %r8
;ALL-NEXT:    pushq %rdi
;ALL-NEXT:    pushq %rsi
;VOID-NEXT:   pushq %rdx
;INT-NEXT:    pushq %rdx
;INT128-NOT:  pushq %rdx
;ALL-NEXT:    pushq %rcx
;VOID-NEXT:   pushq %rax
;INT-NOT:     pushq %rax
;INT128-NOT:  pushq %rax
;ALL-NEXT:    pushq %rbp
;ALL-NEXT:    pushq %r15
;ALL-NEXT:    pushq %r14
;ALL-NEXT:    pushq %r13
;ALL-NEXT:    pushq %r12
;ALL-NEXT:    pushq %rbx
;ALL:         movaps %xmm15
;ALL-NEXT:    movaps %xmm14
;ALL-NEXT:    movaps %xmm13
;ALL-NEXT:    movaps %xmm12
;ALL-NEXT:    movaps %xmm11
;ALL-NEXT:    movaps %xmm10
;ALL-NEXT:    movaps %xmm9
;ALL-NEXT:    movaps %xmm8
;ALL-NEXT:    movaps %xmm7
;ALL-NEXT:    movaps %xmm6
;ALL-NOT:     movaps %xmm5
;ALL-NOT:     movaps %xmm4
;ALL-NOT:     movaps %xmm3
;ALL-NOT:     movaps %xmm2
;ALL-NOT:     movaps %xmm1
;ALL-NOT:     movaps %xmm0
;ALL-NOT:     movaps {{.*}} %xmm0
;ALL-NOT:     movaps {{.*}} %xmm1
;ALL-NOT:     movaps {{.*}} %xmm2
;ALL-NOT:     movaps {{.*}} %xmm3
;ALL-NOT:     movaps {{.*}} %xmm4
;ALL-NOT:     movaps {{.*}} %xmm5
;ALL:         movaps {{.*}} %xmm6
;ALL-NEXT:    movaps {{.*}} %xmm7
;ALL-NEXT:    movaps {{.*}} %xmm8
;ALL-NEXT:    movaps {{.*}} %xmm9
;ALL-NEXT:    movaps {{.*}} %xmm10
;ALL-NEXT:    movaps {{.*}} %xmm11
;ALL-NEXT:    movaps {{.*}} %xmm12
;ALL-NEXT:    movaps {{.*}} %xmm13
;ALL-NEXT:    movaps {{.*}} %xmm14
;ALL-NEXT:    movaps {{.*}} %xmm15
;ALL:         popq    %rbx
;ALL-NEXT:    popq    %r12
;ALL-NEXT:    popq    %r13
;ALL-NEXT:    popq    %r14
;ALL-NEXT:    popq    %r15
;ALL-NEXT:    popq    %rbp
;VOID-NEXT:   popq    %rax
;INT-NOT:     popq    %rax
;INT128-NOT:  popq    %rax
;ALL-NEXT:    popq    %rcx
;VOID-NEXT:   popq    %rdx
;INT-NEXT:    popq    %rdx
;INT128-NOT:  popq    %rdx
;ALL-NEXT:    popq    %rsi
;ALL-NEXT:    popq    %rdi
;ALL-NEXT:    popq    %r8
;ALL-NEXT:    popq    %r9
;ALL-NEXT:    popq    %r10
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret RETTYPE RETVAL
}

; Make sure XMMs are not saved before the call
declare preserve_mostcc RETTYPE @foo(i64, i64, double, double)
define void @preserve_mostcc2() nounwind {
entry:
;ALL-LABEL: preserve_mostcc2
;ALL-NOT:   movaps
;ALL-NOT:   {{.*xmm[0-1,4-9].*}}
  call preserve_mostcc RETTYPE @foo(i64 1, i64 2, double 3.0, double 4.0)
  ret void
}
