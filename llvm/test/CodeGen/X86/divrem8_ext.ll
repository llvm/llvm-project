; RUN: llc -march=x86-64 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-64
; RUN: llc -march=x86    < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-32
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define zeroext i8 @test_udivrem_zext_ah(i8 %x, i8 %y) {
; CHECK-LABEL: test_udivrem_zext_ah
; CHECK:   divb
; CHECK:   movzbl %ah, [[REG_REM:%[a-z0-9]+]]
; CHECK:   movb   %al, ([[REG_ZPTR:%[a-z0-9]+]])
; CHECK:   movl   [[REG_REM]], %eax
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
; CHECK:   movsbl %ah, [[REG_REM:%[a-z0-9]+]]
; CHECK:   movb   %al, ([[REG_ZPTR]])
; CHECK:   movl   [[REG_REM]], %eax
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

@z = external global i8
