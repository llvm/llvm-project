; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

%struct.Large = type { [64 x i32] }

; Check that %rax is also preserved for a function with sret parameter,
define preserve_mostcc void @sret_foo(ptr noalias nocapture writeonly sret(%struct.Large) align 4, i32 noundef) nounwind {
entry:
;CHECK:         pushq %r10
;CHECK-NEXT:    pushq %r9
;CHECK-NEXT:    pushq %r8
;CHECK-NEXT:    pushq %rdi
;CHECK-NEXT:    pushq %rsi
;CHECK-NEXT:    pushq %rdx
;CHECK-NEXT:    pushq %rcx
;CHECK-NEXT:    pushq %rax
;CHECK-NEXT:    pushq %rbp
;CHECK-NEXT:    pushq %r15
;CHECK-NEXT:    pushq %r14
;CHECK-NEXT:    pushq %r13
;CHECK-NEXT:    pushq %r12
;CHECK-NEXT:    pushq %rbx
;CHECK:         popq    %rbx
;CHECK-NEXT:    popq    %r12
;CHECK-NEXT:    popq    %r13
;CHECK-NEXT:    popq    %r14
;CHECK-NEXT:    popq    %r15
;CHECK-NEXT:    popq    %rbp
;CHECK-NEXT:    popq    %rax
;CHECK-NEXT:    popq    %rcx
;CHECK-NEXT:    popq    %rdx
;CHECK-NEXT:    popq    %rsi
;CHECK-NEXT:    popq    %rdi
;CHECK-NEXT:    popq    %r8
;CHECK-NEXT:    popq    %r9
;CHECK-NEXT:    popq    %r10
  call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{rbp},~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  ret void
}

; Check that neither %rax no %rdi are caller-saved.
define void @foo(ptr noalias nocapture writeonly sret(%struct.Large) align 4 %0, i32 noundef %1) nounwind {
entry:
;CHECK-NOT:   movq %rax, [[REG1:%[a-z0-9]+]]
;CHECK-NOT:   movq %rdi, [[REG2:%[a-z0-9]+]]
;CHECK:       movq %r11, [[REG3:%[a-z0-9]+]]
;CHECK:       movaps %xmm2
;CHECK:       movaps %xmm3
;CHECK:       movaps %xmm4
;CHECK:       movaps %xmm5
;CHECK:       movaps %xmm6
;CHECK:       movaps %xmm7
;CHECK:       movaps %xmm8
;CHECK:       movaps %xmm9
;CHECK:       movaps %xmm10
;CHECK:       movaps %xmm11
;CHECK:       movaps %xmm12
;CHECK:       movaps %xmm13
;CHECK:       movaps %xmm14
;CHECK:       movaps %xmm15
;CHECK:       call
;CHECK-NOT:   movq {{.*}}, %rax
;CHECK-NOT:   movq {{.*}}, %rdi
;CHECK:       movq [[REG3]], %r11
;CHECK:       movaps {{.*}} %xmm2
;CHECK:       movaps {{.*}} %xmm3
;CHECK:       movaps {{.*}} %xmm4
;CHECK:       movaps {{.*}} %xmm5
;CHECK:       movaps {{.*}} %xmm6
;CHECK:       movaps {{.*}} %xmm7
;CHECK:       movaps {{.*}} %xmm8
;CHECK:       movaps {{.*}} %xmm9
;CHECK:       movaps {{.*}} %xmm10
;CHECK:       movaps {{.*}} %xmm11
;CHECK:       movaps {{.*}} %xmm12
;CHECK:       movaps {{.*}} %xmm13
;CHECK:       movaps {{.*}} %xmm14
;CHECK:       movaps {{.*}} %xmm15
  %a0 = call i64 asm sideeffect "", "={rax}"() nounwind
  %a1 = call i64 asm sideeffect "", "={rcx}"() nounwind
  %a2 = call i64 asm sideeffect "", "={rdx}"() nounwind
  %a3 = call i64 asm sideeffect "", "={r8}"() nounwind
  %a4 = call i64 asm sideeffect "", "={r9}"() nounwind
  %a5 = call i64 asm sideeffect "", "={r10}"() nounwind
  %a6 = call i64 asm sideeffect "", "={r11}"() nounwind
  %a10 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
  %a11 = call <2 x double> asm sideeffect "", "={xmm3}"() nounwind
  %a12 = call <2 x double> asm sideeffect "", "={xmm4}"() nounwind
  %a13 = call <2 x double> asm sideeffect "", "={xmm5}"() nounwind
  %a14 = call <2 x double> asm sideeffect "", "={xmm6}"() nounwind
  %a15 = call <2 x double> asm sideeffect "", "={xmm7}"() nounwind
  %a16 = call <2 x double> asm sideeffect "", "={xmm8}"() nounwind
  %a17 = call <2 x double> asm sideeffect "", "={xmm9}"() nounwind
  %a18 = call <2 x double> asm sideeffect "", "={xmm10}"() nounwind
  %a19 = call <2 x double> asm sideeffect "", "={xmm11}"() nounwind
  %a20 = call <2 x double> asm sideeffect "", "={xmm12}"() nounwind
  %a21 = call <2 x double> asm sideeffect "", "={xmm13}"() nounwind
  %a22 = call <2 x double> asm sideeffect "", "={xmm14}"() nounwind
  %a23 = call <2 x double> asm sideeffect "", "={xmm15}"() nounwind
  call preserve_mostcc void @sret_foo(ptr sret(%struct.Large) align 4 %0, i32 noundef %1)
  call void asm sideeffect "", "{rax},{rcx},{rdx},{r8},{r9},{r10},{r11},{xmm2},{xmm3},{xmm4},{xmm5},{xmm6},{xmm7},{xmm8},{xmm9},{xmm10},{xmm11},{xmm12},{xmm13},{xmm14},{xmm15}"(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, <2 x double> %a10, <2 x double> %a11, <2 x double> %a12, <2 x double> %a13, <2 x double> %a14, <2 x double> %a15, <2 x double> %a16, <2 x double> %a17, <2 x double> %a18, <2 x double> %a19, <2 x double> %a20, <2 x double> %a21, <2 x double> %a22, <2 x double> %a23)
  ret void
}
