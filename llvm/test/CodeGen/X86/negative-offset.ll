; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test that a constant consisting of a global symbol with a negative offset
; is properly folded and isel'd.

; CHECK-LABEL: negative_offset:
; CHECK: movl   $G, %eax
; CHECK: notq   %rax
; CHECK: addq   %rdi, %rax
; CHECK: retq
@G = external global [8 x i32]
define i8* @negative_offset(i8* %a) {
  %t = getelementptr i8, i8* %a, i64 sub (i64 -1, i64 ptrtoint ([8 x i32]* @G to i64))
  ret i8* %t
}
