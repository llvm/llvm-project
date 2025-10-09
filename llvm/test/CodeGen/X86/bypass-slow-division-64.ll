; Check that 64-bit division is bypassed correctly.
; RUN: llc < %s -mtriple=x86_64-- -mattr=-idivq-to-divl | FileCheck %s --check-prefixes=CHECK,FAST-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mattr=+idivq-to-divl | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64-v2       | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64-v3       | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64-v4       | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; Intel
; RUN: llc < %s -mtriple=x86_64-- -mcpu=nehalem         | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=sandybridge     | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=haswell         | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=skylake         | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=alderlake       | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; AMD
; RUN: llc < %s -mtriple=x86_64-- -mcpu=barcelona       | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=btver1          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=btver2          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver1          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver2          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver3          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=bdver4          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver1          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver2          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver3          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver4          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver5          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ

; Additional tests for 64-bit divide bypass

;
; SDIV
;

define i64 @sdiv_quotient(i64 %a, i64 %b) nounwind {
; FAST-DIVQ-LABEL: sdiv_quotient:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    cqto
; FAST-DIVQ-NEXT:    idivq %rsi
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: sdiv_quotient:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movq %rdi, %rcx
; SLOW-DIVQ-DAG:     orq %rsi, %rcx
; SLOW-DIVQ-DAG:     shrq $32, %rcx
; SLOW-DIVQ-NEXT:    je .LBB0_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    cqto
; SLOW-DIVQ-NEXT:    idivq %rsi
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB0_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    retq
  %result = sdiv i64 %a, %b
  ret i64 %result
}

define i64 @sdiv_quotient_optsize(i64 %a, i64 %b) nounwind optsize {
; CHECK-LABEL: sdiv_quotient_optsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    cqto
; CHECK-NEXT:    idivq %rsi
; CHECK-NEXT:    retq
  %result = sdiv i64 %a, %b
  ret i64 %result
}

define i64 @sdiv_quotient_minsize(i64 %a, i64 %b) nounwind minsize {
; CHECK-LABEL: sdiv_quotient_minsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    cqto
; CHECK-NEXT:    idivq %rsi
; CHECK-NEXT:    retq
  %result = sdiv i64 %a, %b
  ret i64 %result
}

define i64 @sdiv_remainder(i64 %a, i64 %b) nounwind {
; FAST-DIVQ-LABEL: sdiv_remainder:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    cqto
; FAST-DIVQ-NEXT:    idivq %rsi
; FAST-DIVQ-NEXT:    movq %rdx, %rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: sdiv_remainder:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movq %rdi, %rcx
; SLOW-DIVQ-DAG:     orq %rsi, %rcx
; SLOW-DIVQ-DAG:     shrq $32, %rcx
; SLOW-DIVQ-NEXT:    je .LBB3_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    cqto
; SLOW-DIVQ-NEXT:    idivq %rsi
; SLOW-DIVQ-NEXT:    movq %rdx, %rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB3_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    movl %edx, %eax
; SLOW-DIVQ-NEXT:    retq
  %result = srem i64 %a, %b
  ret i64 %result
}

define i64 @sdiv_remainder_optsize(i64 %a, i64 %b) nounwind optsize {
; CHECK-LABEL: sdiv_remainder_optsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    cqto
; CHECK-NEXT:    idivq %rsi
; CHECK-NEXT:    movq %rdx, %rax
; CHECK-NEXT:    retq
  %result = srem i64 %a, %b
  ret i64 %result
}

define i64 @sdiv_remainder_minsize(i64 %a, i64 %b) nounwind minsize {
; CHECK-LABEL: sdiv_remainder_minsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    cqto
; CHECK-NEXT:    idivq %rsi
; CHECK-NEXT:    movq %rdx, %rax
; CHECK-NEXT:    retq
  %result = srem i64 %a, %b
  ret i64 %result
}

define i64 @sdiv_quotient_and_remainder(i64 %a, i64 %b) nounwind {
; FAST-DIVQ-LABEL: sdiv_quotient_and_remainder:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    cqto
; FAST-DIVQ-NEXT:    idivq %rsi
; FAST-DIVQ-NEXT:    addq %rdx, %rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: sdiv_quotient_and_remainder:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movq %rdi, %rcx
; SLOW-DIVQ-DAG:     orq %rsi, %rcx
; SLOW-DIVQ-DAG:     shrq $32, %rcx
; SLOW-DIVQ-NEXT:    je .LBB6_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    cqto
; SLOW-DIVQ-NEXT:    idivq %rsi
; SLOW-DIVQ-NEXT:    addq %rdx, %rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB6_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    # kill: def $edx killed $edx def $rdx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    addq %rdx, %rax
; SLOW-DIVQ-NEXT:    retq
  %resultdiv = sdiv i64 %a, %b
  %resultrem = srem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}

define i64 @sdiv_quotient_and_remainder_optsize(i64 %a, i64 %b) nounwind optsize {
; CHECK-LABEL: sdiv_quotient_and_remainder_optsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    cqto
; CHECK-NEXT:    idivq %rsi
; CHECK-NEXT:    addq %rdx, %rax
; CHECK-NEXT:    retq
  %resultdiv = sdiv i64 %a, %b
  %resultrem = srem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}

define i64 @sdiv_quotient_and_remainder_minsize(i64 %a, i64 %b) nounwind minsize {
; CHECK-LABEL: sdiv_quotient_and_remainder_minsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    cqto
; CHECK-NEXT:    idivq %rsi
; CHECK-NEXT:    addq %rdx, %rax
; CHECK-NEXT:    retq
  %resultdiv = sdiv i64 %a, %b
  %resultrem = srem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}

;
; UDIV
;

define i64 @udiv_quotient(i64 %a, i64 %b) nounwind {
; FAST-DIVQ-LABEL: udiv_quotient:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    xorl %edx, %edx
; FAST-DIVQ-NEXT:    divq %rsi
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: udiv_quotient:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movq %rdi, %rcx
; SLOW-DIVQ-DAG:     orq %rsi, %rcx
; SLOW-DIVQ-DAG:     shrq $32, %rcx
; SLOW-DIVQ-NEXT:    je .LBB9_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divq %rsi
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB9_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    retq
  %result = udiv i64 %a, %b
  ret i64 %result
}

define i64 @udiv_quotient_optsize(i64 %a, i64 %b) nounwind optsize {
; CHECK-LABEL: udiv_quotient_optsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divq %rsi
; CHECK-NEXT:    retq
  %result = udiv i64 %a, %b
  ret i64 %result
}

define i64 @udiv_quotient_minsize(i64 %a, i64 %b) nounwind minsize {
; CHECK-LABEL: udiv_quotient_minsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divq %rsi
; CHECK-NEXT:    retq
  %result = udiv i64 %a, %b
  ret i64 %result
}

define i64 @udiv_remainder(i64 %a, i64 %b) nounwind {
; FAST-DIVQ-LABEL: udiv_remainder:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    xorl %edx, %edx
; FAST-DIVQ-NEXT:    divq %rsi
; FAST-DIVQ-NEXT:    movq %rdx, %rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: udiv_remainder:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movq %rdi, %rcx
; SLOW-DIVQ-DAG:     orq %rsi, %rcx
; SLOW-DIVQ-DAG:     shrq $32, %rcx
; SLOW-DIVQ-NEXT:    je .LBB12_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divq %rsi
; SLOW-DIVQ-NEXT:    movq %rdx, %rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB12_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    movl %edx, %eax
; SLOW-DIVQ-NEXT:    retq
  %result = urem i64 %a, %b
  ret i64 %result
}

define i64 @udiv_remainder_optsize(i64 %a, i64 %b) nounwind optsize {
; CHECK-LABEL: udiv_remainder_optsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divq %rsi
; CHECK-NEXT:    movq %rdx, %rax
; CHECK-NEXT:    retq
  %result = urem i64 %a, %b
  ret i64 %result
}

define i64 @udiv_remainder_minsize(i64 %a, i64 %b) nounwind minsize {
; CHECK-LABEL: udiv_remainder_minsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divq %rsi
; CHECK-NEXT:    movq %rdx, %rax
; CHECK-NEXT:    retq
  %result = urem i64 %a, %b
  ret i64 %result
}

define i64 @udiv_quotient_and_remainder(i64 %a, i64 %b) nounwind {
; FAST-DIVQ-LABEL: udiv_quotient_and_remainder:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    xorl %edx, %edx
; FAST-DIVQ-NEXT:    divq %rsi
; FAST-DIVQ-NEXT:    addq %rdx, %rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: udiv_quotient_and_remainder:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movq %rdi, %rcx
; SLOW-DIVQ-DAG:     orq %rsi, %rcx
; SLOW-DIVQ-DAG:     shrq $32, %rcx
; SLOW-DIVQ-NEXT:    je .LBB15_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divq %rsi
; SLOW-DIVQ-NEXT:    addq %rdx, %rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB15_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    # kill: def $edx killed $edx def $rdx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    addq %rdx, %rax
; SLOW-DIVQ-NEXT:    retq
  %resultdiv = udiv i64 %a, %b
  %resultrem = urem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}

define i64 @udiv_quotient_and_remainder_optsize(i64 %a, i64 %b) nounwind optsize {
; CHECK-LABEL: udiv_quotient_and_remainder_optsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divq %rsi
; CHECK-NEXT:    addq %rdx, %rax
; CHECK-NEXT:    retq
  %resultdiv = udiv i64 %a, %b
  %resultrem = urem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}

define i64 @udiv_quotient_and_remainder_minsize(i64 %a, i64 %b) nounwind minsize {
; CHECK-LABEL: udiv_quotient_and_remainder_minsize:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divq %rsi
; CHECK-NEXT:    addq %rdx, %rax
; CHECK-NEXT:    retq
  %resultdiv = udiv i64 %a, %b
  %resultrem = urem i64 %a, %b
  %result = add i64 %resultdiv, %resultrem
  ret i64 %result
}

define void @PR43514(i32 %x, i32 %y) {
; CHECK-LABEL: PR43514:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %z1 = zext i32 %x to i64
  %z2 = zext i32 %y to i64
  %s = srem i64 %z1, %z2
  ret void
}
