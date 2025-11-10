; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; This test checks that a pattern of two 32-bit loads, which are combined
; to form a 64-bit value with swapped words, is optimized into a single
; 64-bit load followed by a 32-bit rotate.

define i64 @test_load_bswap_to_rotate(ptr %p) {
; CHECK-LABEL: test_load_bswap_to_rotate:
; CHECK:  # %bb.0:
; CHECK-NEXT:    movq (%rdi), %rax
; CHECK-NEXT:    rorq $32, %rax
; CHECK-NEXT:    retq
;
; CHECK-NOT: movl

  %p.hi = getelementptr inbounds nuw i8, ptr %p, i64 4
  %lo = load i32, ptr %p
  %hi = load i32, ptr %p.hi
  %conv = zext i32 %lo to i64
  %shl = shl nuw i64 %conv, 32
  %conv2 = zext i32 %hi to i64
  %or = or disjoint i64 %shl, %conv2
  ret i64 %or
}
