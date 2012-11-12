; RUN: llc < %s | FileCheck %s

; Check that the shr(shl X, 56), 48) is not mistakenly turned into
; a shr (X, -8) that gets subsequently "optimized away" as undef
; PR4254

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @foo(i64 %b) nounwind readnone {
entry:
; CHECK: foo:
; CHECK: shlq $56, %rdi
; CHECK: sarq $48, %rdi
; CHECK: leaq 1(%rdi), %rax
	%shl = shl i64 %b, 56		; <i64> [#uses=1]
	%shr = ashr i64 %shl, 48		; <i64> [#uses=1]
	%add5 = or i64 %shr, 1		; <i64> [#uses=1]
	ret i64 %add5
}
