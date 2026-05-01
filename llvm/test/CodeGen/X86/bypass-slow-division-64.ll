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
; RUN: llc < %s -mtriple=x86_64-- -mcpu=znver6          | FileCheck %s --check-prefixes=CHECK,SLOW-DIVQ

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

; dividend > U32_MAX, bypass emits divq only
define i32 @udiv_i64_i32_assume_dividend_gt_u32_max(i64 %n, i32 %d) {
; FAST-DIVQ-LABEL: udiv_i64_i32_assume_dividend_gt_u32_max:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    movl %esi, %ecx
; FAST-DIVQ-NEXT:    xorl %edx, %edx
; FAST-DIVQ-NEXT:    divq %rcx
; FAST-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: udiv_i64_i32_assume_dividend_gt_u32_max:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movl %esi, %ecx
; SLOW-DIVQ-DAG:     movq %rdi, %rdx
; SLOW-DIVQ-DAG:     shrq $32, %rdx
; SLOW-DIVQ-NEXT:    je .LBB19_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divq %rcx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB19_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %ecx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
  %cmp = icmp ugt i64 %n, 4294967295
  call void @llvm.assume(i1 %cmp)
  %d.ext = zext i32 %d to i64
  %q = udiv i64 %n, %d.ext
  %tr = trunc i64 %q to i32
  ret i32 %tr
}

; dividend > U32_MAX, bypass emits divq only
define i32 @urem_i64_i32_assume_dividend_gt_u32_max(i64 %n, i32 %d) {
; FAST-DIVQ-LABEL: urem_i64_i32_assume_dividend_gt_u32_max:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    movl %esi, %ecx
; FAST-DIVQ-NEXT:    xorl %edx, %edx
; FAST-DIVQ-NEXT:    divq %rcx
; FAST-DIVQ-NEXT:    movq %rdx, %rax
; FAST-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: urem_i64_i32_assume_dividend_gt_u32_max:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movl %esi, %ecx
; SLOW-DIVQ-DAG:     movq %rdi, %rdx
; SLOW-DIVQ-DAG:     shrq $32, %rdx
; SLOW-DIVQ-NEXT:    je .LBB20_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divq %rcx
; SLOW-DIVQ-NEXT:    movq %rdx, %rax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB20_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %ecx
; SLOW-DIVQ-NEXT:    movl %edx, %eax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
  %cmp = icmp ugt i64 %n, 4294967295
  call void @llvm.assume(i1 %cmp)
  %d.ext = zext i32 %d to i64
  %r = urem i64 %n, %d.ext
  %tr = trunc i64 %r to i32
  ret i32 %tr
}

; dividend > U32_MAX, bypass emits idivq only
define i32 @sdiv_i64_i32_assume_dividend_gt_u32_max(i64 %n, i32 %d) {
; FAST-DIVQ-LABEL: sdiv_i64_i32_assume_dividend_gt_u32_max:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    movslq %esi, %rcx
; FAST-DIVQ-NEXT:    cqto
; FAST-DIVQ-NEXT:    idivq %rcx
; FAST-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: sdiv_i64_i32_assume_dividend_gt_u32_max:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movslq %esi, %rcx
; SLOW-DIVQ-DAG:     movq %rdi, %rdx
; SLOW-DIVQ-DAG:     orq %rcx, %rdx
; SLOW-DIVQ-DAG:     shrq $32, %rdx
; SLOW-DIVQ-NEXT:    je .LBB21_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    cqto
; SLOW-DIVQ-NEXT:    idivq %rcx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB21_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
  %cmp = icmp sgt i64 %n, 4294967295
  call void @llvm.assume(i1 %cmp)
  %d.ext = sext i32 %d to i64
  %q = sdiv i64 %n, %d.ext
  %tr = trunc i64 %q to i32
  ret i32 %tr
}

; dividend > U32_MAX, bypass emits idivq only
define i32 @srem_i64_i32_assume_dividend_gt_u32_max(i64 %n, i32 %d) {
; FAST-DIVQ-LABEL: srem_i64_i32_assume_dividend_gt_u32_max:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    movslq %esi, %rcx
; FAST-DIVQ-NEXT:    cqto
; FAST-DIVQ-NEXT:    idivq %rcx
; FAST-DIVQ-NEXT:    movq %rdx, %rax
; FAST-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: srem_i64_i32_assume_dividend_gt_u32_max:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movslq %esi, %rcx
; SLOW-DIVQ-DAG:     movq %rdi, %rdx
; SLOW-DIVQ-DAG:     orq %rcx, %rdx
; SLOW-DIVQ-DAG:     shrq $32, %rdx
; SLOW-DIVQ-NEXT:    je .LBB22_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    cqto
; SLOW-DIVQ-NEXT:    idivq %rcx
; SLOW-DIVQ-NEXT:    movq %rdx, %rax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB22_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    movl %edx, %eax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
  %cmp = icmp sgt i64 %n, 4294967295
  call void @llvm.assume(i1 %cmp)
  %d.ext = sext i32 %d to i64
  %r = srem i64 %n, %d.ext
  %tr = trunc i64 %r to i32
  ret i32 %tr
}

; nonzero-divisor assumption carries no width fact, udiv branch still emitted
define i32 @udiv_i64_i32_assume_divisor_nonzero_no_width_fact(i64 %n, i32 %d) {
; FAST-DIVQ-LABEL: udiv_i64_i32_assume_divisor_nonzero_no_width_fact:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    movl %esi, %ecx
; FAST-DIVQ-NEXT:    xorl %edx, %edx
; FAST-DIVQ-NEXT:    divq %rcx
; FAST-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: udiv_i64_i32_assume_divisor_nonzero_no_width_fact:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movl %esi, %ecx
; SLOW-DIVQ-DAG:     movq %rdi, %rdx
; SLOW-DIVQ-DAG:     shrq $32, %rdx
; SLOW-DIVQ-NEXT:    je .LBB23_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divq %rcx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB23_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %ecx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
  %cmp = icmp ne i32 %d, 0
  call void @llvm.assume(i1 %cmp)
  %d.ext = zext i32 %d to i64
  %q = udiv i64 %n, %d.ext
  %tr = trunc i64 %q to i32
  ret i32 %tr
}

; nonzero-divisor assumption carries no width fact, sdiv branch still emitted
define i32 @sdiv_i64_i32_assume_divisor_nonzero_no_width_fact(i64 %n, i32 %d) {
; FAST-DIVQ-LABEL: sdiv_i64_i32_assume_divisor_nonzero_no_width_fact:
; FAST-DIVQ:       # %bb.0:
; FAST-DIVQ-NEXT:    movq %rdi, %rax
; FAST-DIVQ-NEXT:    movslq %esi, %rcx
; FAST-DIVQ-NEXT:    cqto
; FAST-DIVQ-NEXT:    idivq %rcx
; FAST-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; FAST-DIVQ-NEXT:    retq
;
; SLOW-DIVQ-LABEL: sdiv_i64_i32_assume_divisor_nonzero_no_width_fact:
; SLOW-DIVQ:       # %bb.0:
; SLOW-DIVQ-DAG:     movq %rdi, %rax
; SLOW-DIVQ-DAG:     movslq %esi, %rcx
; SLOW-DIVQ-DAG:     movq %rdi, %rdx
; SLOW-DIVQ-DAG:     orq %rcx, %rdx
; SLOW-DIVQ-DAG:     shrq $32, %rdx
; SLOW-DIVQ-NEXT:    je .LBB24_1
; SLOW-DIVQ-NEXT:  # %bb.2:
; SLOW-DIVQ-NEXT:    cqto
; SLOW-DIVQ-NEXT:    idivq %rcx
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
; SLOW-DIVQ-NEXT:  .LBB24_1:
; SLOW-DIVQ-DAG:     # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-DAG:     xorl %edx, %edx
; SLOW-DIVQ-NEXT:    divl %esi
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax def $rax
; SLOW-DIVQ-NEXT:    # kill: def $eax killed $eax killed $rax
; SLOW-DIVQ-NEXT:    retq
  %cmp = icmp ne i32 %d, 0
  call void @llvm.assume(i1 %cmp)
  %d.ext = sext i32 %d to i64
  %q = sdiv i64 %n, %d.ext
  %tr = trunc i64 %q to i32
  ret i32 %tr
}
