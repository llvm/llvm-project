; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -machine-combiner-verify-pattern-order=true | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -stop-after machine-combiner -machine-combiner-verify-pattern-order=true -o - | FileCheck %s --check-prefix=DEAD

; Verify that integer multiplies are reassociated. The first multiply in 
; each test should be independent of the result of the preceding add (lea).

; TODO: This test does not actually test i16 machine instruction reassociation 
; because the operands are being promoted to i32 types.

define i16 @reassociate_muls_i16(i16 %x0, i16 %x1, i16 %x2, i16 %x3) {
; CHECK-LABEL: reassociate_muls_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill
; CHECK-NEXT:    # kill
; CHECK-NEXT:    leal   (%rdi,%rsi), %eax
; CHECK-NEXT:    imull  %edx, %eax
; CHECK-NEXT:    imull  %ecx, %eax
; CHECK-NEXT:    # kill
; CHECK-NEXT:    retq
  %t0 = add i16 %x0, %x1
  %t1 = mul i16 %x2, %t0
  %t2 = mul i16 %x3, %t1
  ret i16 %t2
}

define i32 @reassociate_muls_i32(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
; CHECK-LABEL: reassociate_muls_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill
; CHECK-NEXT:    # kill
; CHECK-NEXT:    leal   (%rdi,%rsi), %eax
; CHECK-NEXT:    imull  %edx, %eax
; CHECK-NEXT:    imull  %ecx, %eax
; CHECK-NEXT:    retq

; DEAD:       ADD32rr
; DEAD-NEXT:  IMUL32rr{{.*}}implicit-def dead $eflags
; DEAD-NEXT:  IMUL32rr{{.*}}implicit-def dead $eflags

  %t0 = add i32 %x0, %x1
  %t1 = mul i32 %x2, %t0
  %t2 = mul i32 %x3, %t1
  ret i32 %t2
}

define i64 @reassociate_muls_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3) {
; CHECK-LABEL: reassociate_muls_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    leaq   (%rdi,%rsi), %rax
; CHECK-NEXT:    imulq  %rdx, %rax
; CHECK-NEXT:    imulq  %rcx, %rax
; CHECK-NEXT:    retq
  %t0 = add i64 %x0, %x1
  %t1 = mul i64 %x2, %t0
  %t2 = mul i64 %x3, %t1
  ret i64 %t2
}

; Verify that integer 'ands' are reassociated. The first 'and' in 
; each test should be independent of the result of the preceding sub.

define i8 @reassociate_ands_i8(i8 %x0, i8 %x1, i8 %x2, i8 %x3) {
; CHECK-LABEL: reassociate_ands_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl  %edi, %eax
; CHECK-NEXT:    subb  %sil, %al
; CHECK-NEXT:    andb  %dl, %al
; CHECK-NEXT:    andb  %cl, %al
; CHECK-NEXT:    # kill
; CHECK-NEXT:    retq
  %t0 = sub i8 %x0, %x1
  %t1 = and i8 %x2, %t0
  %t2 = and i8 %x3, %t1
  ret i8 %t2
}

; TODO: No way to test i16? These appear to always get promoted to i32.

define i32 @reassociate_ands_i32(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
; CHECK-LABEL: reassociate_ands_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl  %edi, %eax
; CHECK-NEXT:    subl  %esi, %eax
; CHECK-NEXT:    andl  %edx, %eax
; CHECK-NEXT:    andl  %ecx, %eax
; CHECK-NEXT:    retq
  %t0 = sub i32 %x0, %x1
  %t1 = and i32 %x2, %t0
  %t2 = and i32 %x3, %t1
  ret i32 %t2
}

define i64 @reassociate_ands_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3) {
; CHECK-LABEL: reassociate_ands_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq  %rdi, %rax
; CHECK-NEXT:    subq  %rsi, %rax
; CHECK-NEXT:    andq  %rdx, %rax
; CHECK-NEXT:    andq  %rcx, %rax
; CHECK-NEXT:    retq
  %t0 = sub i64 %x0, %x1
  %t1 = and i64 %x2, %t0
  %t2 = and i64 %x3, %t1
  ret i64 %t2
}

; Verify that integer 'ors' are reassociated. The first 'or' in 
; each test should be independent of the result of the preceding sub.

define i8 @reassociate_ors_i8(i8 %x0, i8 %x1, i8 %x2, i8 %x3) {
; CHECK-LABEL: reassociate_ors_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl  %edi, %eax
; CHECK-NEXT:    subb  %sil, %al
; CHECK-NEXT:    orb   %dl, %al
; CHECK-NEXT:    orb   %cl, %al
; CHECK-NEXT:    # kill
; CHECK-NEXT:    retq
  %t0 = sub i8 %x0, %x1
  %t1 = or i8 %x2, %t0
  %t2 = or i8 %x3, %t1
  ret i8 %t2
}

; TODO: No way to test i16? These appear to always get promoted to i32.

define i32 @reassociate_ors_i32(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
; CHECK-LABEL: reassociate_ors_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl  %edi, %eax
; CHECK-NEXT:    subl  %esi, %eax
; CHECK-NEXT:    orl   %edx, %eax
; CHECK-NEXT:    orl   %ecx, %eax
; CHECK-NEXT:    retq
  %t0 = sub i32 %x0, %x1
  %t1 = or i32 %x2, %t0
  %t2 = or i32 %x3, %t1
  ret i32 %t2
}

define i64 @reassociate_ors_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3) {
; CHECK-LABEL: reassociate_ors_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq  %rdi, %rax
; CHECK-NEXT:    subq  %rsi, %rax
; CHECK-NEXT:    orq   %rdx, %rax
; CHECK-NEXT:    orq   %rcx, %rax
; CHECK-NEXT:    retq
  %t0 = sub i64 %x0, %x1
  %t1 = or i64 %x2, %t0
  %t2 = or i64 %x3, %t1
  ret i64 %t2
}

; Verify that integer 'xors' are reassociated. The first 'xor' in
; each test should be independent of the result of the preceding sub.

define i8 @reassociate_xors_i8(i8 %x0, i8 %x1, i8 %x2, i8 %x3) {
; CHECK-LABEL: reassociate_xors_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl  %edi, %eax
; CHECK-NEXT:    subb  %sil, %al
; CHECK-NEXT:    xorb  %dl, %al
; CHECK-NEXT:    xorb  %cl, %al
; CHECK-NEXT:    # kill
; CHECK-NEXT:    retq
  %t0 = sub i8 %x0, %x1
  %t1 = xor i8 %x2, %t0
  %t2 = xor i8 %x3, %t1
  ret i8 %t2
}

; TODO: No way to test i16? These appear to always get promoted to i32.

define i32 @reassociate_xors_i32(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
; CHECK-LABEL: reassociate_xors_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl  %edi, %eax
; CHECK-NEXT:    subl  %esi, %eax
; CHECK-NEXT:    xorl  %edx, %eax
; CHECK-NEXT:    xorl  %ecx, %eax
; CHECK-NEXT:    retq
  %t0 = sub i32 %x0, %x1
  %t1 = xor i32 %x2, %t0
  %t2 = xor i32 %x3, %t1
  ret i32 %t2
}

define i64 @reassociate_xors_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3) {
; CHECK-LABEL: reassociate_xors_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq  %rdi, %rax
; CHECK-NEXT:    subq  %rsi, %rax
; CHECK-NEXT:    xorq  %rdx, %rax
; CHECK-NEXT:    xorq  %rcx, %rax
; CHECK-NEXT:    retq
  %t0 = sub i64 %x0, %x1
  %t1 = xor i64 %x2, %t0
  %t2 = xor i64 %x3, %t1
  ret i64 %t2
}

