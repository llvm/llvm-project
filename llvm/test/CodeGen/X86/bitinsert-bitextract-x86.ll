; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define i16 @test_bitextract_var(b32 %src, i32 %off) {
; CHECK-LABEL: test_bitextract_var:
; CHECK:       movl %edi, %eax
; CHECK:       leal 16(%rsi), %ecx
; CHECK:       rorl %cl, %eax
; CHECK:       retq
  %result = bitextract i16, b32 %src, i32 %off
  ret i16 %result
}

define i16 @test_bitextract_const(b32 %src) {
; CHECK-LABEL: test_bitextract_const:
; CHECK:       movl %edi, %eax
; CHECK:       rorl $24, %eax
; CHECK:       retq
  %result = bitextract i16, b32 %src, i32 8
  ret i16 %result
}

define i8 @test_bitextract_narrow(b64 %src) {
; CHECK-LABEL: test_bitextract_narrow:
; CHECK:       movq %rdi, %rax
; CHECK:       shrl $8, %eax
; CHECK:       retq
  %result = bitextract i8, b64 %src, i32 0
  ret i8 %result
}

define b32 @test_bitinsert_var(b32 %base, i16 %val, i32 %off) {
; CHECK-LABEL: test_bitinsert_var:
; CHECK:       leal 16(%rdx), %ecx
; CHECK:       rorl %cl, %edi
; CHECK:       andl $-65536, %edi
; CHECK:       movzwl %si, %eax
; CHECK:       orl %edi, %eax
; CHECK:       roll %cl, %eax
; CHECK:       retq
  %result = bitinsert b32 %base, i16 %val, i32 %off
  ret b32 %result
}
