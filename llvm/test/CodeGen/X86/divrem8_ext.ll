; RUN: llc -march=x86-64 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-64
; RUN: llc -march=x86    < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-32
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define zeroext i8 @test_udivrem_zext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_udivrem_zext_ah
; CHECK:   divb
; CHECK:   movzbl %ah, %e[[REG_REM:[a-z]]]x
; CHECK:   movb   %al, ([[REG_ZPTR:%[a-z0-9]+]])
; CHECK:   movb   %[[REG_REM]]l, %al
; CHECK:   ret
  %div = udiv i8 %x, %y
  store i8 %div, i8* @z
  %1 = urem i8 %x, %y
  ret i8 %1
}

define zeroext i8 @test_urem_zext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_urem_zext_ah
; CHECK:   divb
; CHECK:   movzbl %ah, %eax
; CHECK:   ret
  %1 = urem i8 %x, %y
  ret i8 %1
}

define i8 @test_urem_noext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_urem_noext_ah
; CHECK:   divb   [[REG_X:%[a-z0-9]+]]
; CHECK:   movzbl %ah, %eax
; CHECK:   addb   [[REG_X]], %al
; CHECK:   ret
  %1 = urem i8 %x, %y
  %2 = add i8 %1, %y
  ret i8 %2
}

define i64 @test_urem_zext64_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_urem_zext64_ah
; CHECK:    divb
; CHECK:    movzbl %ah, %eax
; CHECK-32: xorl %edx, %edx
; CHECK:    ret
  %1 = urem i8 %x, %y
  %2 = zext i8 %1 to i64
  ret i64 %2
}

define signext i8 @test_sdivrem_sext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_sdivrem_sext_ah
; CHECK:   cbtw
; CHECK:   idivb
; CHECK:   movsbl %ah, %e[[REG_REM:[a-z]]]x
; CHECK:   movb   %al, ([[REG_ZPTR]])
; CHECK:   movb   %[[REG_REM]]l, %al
; CHECK:   ret
  %div = sdiv i8 %x, %y
  store i8 %div, i8* @z
  %1 = srem i8 %x, %y
  ret i8 %1
}

define signext i8 @test_srem_sext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_srem_sext_ah
; CHECK:   cbtw
; CHECK:   idivb
; CHECK:   movsbl %ah, %eax
; CHECK:   ret
  %1 = srem i8 %x, %y
  ret i8 %1
}

define i8 @test_srem_noext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_srem_noext_ah
; CHECK:   cbtw
; CHECK:   idivb [[REG_X:%[a-z0-9]+]]
; CHECK:   movsbl %ah, %eax
; CHECK:   addb   [[REG_X]], %al
; CHECK:   ret
  %1 = srem i8 %x, %y
  %2 = add i8 %1, %y
  ret i8 %2
}

define i64 @test_srem_sext64_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_srem_sext64_ah
; CHECK:    cbtw
; CHECK:    idivb
; CHECK:    movsbl %ah, %eax
; CHECK-32: movl %eax, %edx
; CHECK-32: sarl $31, %edx
; CHECK-64: movsbq %al, %rax
; CHECK:    ret
  %1 = srem i8 %x, %y
  %2 = sext i8 %1 to i64
  ret i64 %2
}

define i64 @pr25754(i8 %a, i8 %c) {
; CHECK-LABEL: pr25754
; CHECK:    movzbl {{.+}}, %eax
; CHECK:    divb
; CHECK:    movzbl %ah, %ecx
; CHECK:    movzbl %al, %eax
; CHECK-32: addl %ecx, %eax
; CHECK-32: sbbl %edx, %edx
; CHECK-32: andl $1, %edx
; CHECK-64: addq %rcx, %rax
; CHECK:    ret
  %r1 = urem i8 %a, %c
  %d1 = udiv i8 %a, %c
  %r2 = zext i8 %r1 to i64
  %d2 = zext i8 %d1 to i64
  %ret = add i64 %r2, %d2
  ret i64 %ret
}

@z = external global i8
