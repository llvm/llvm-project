; RUN: llc -no-integrated-as < %s | FileCheck %s
; ModuleID = '4964.c'
; PR 4964
; Registers other than RAX, RCX are OK, but they must be different.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"
	%0 = type { i64, i64 }		; type %0

define i64 @flsst(i64 %find) nounwind ssp {
entry:
; CHECK: FOO %rax %rcx
	%asmtmp = tail call %0 asm sideeffect "FOO $0 $1 $2", "=r,=&r,rm,~{dirflag},~{fpsr},~{flags},~{cc}"(i64 %find) nounwind		; <%0> [#uses=1]
	%asmresult = extractvalue %0 %asmtmp, 0		; <i64> [#uses=1]
	ret i64 %asmresult
}
