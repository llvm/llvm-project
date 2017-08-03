; RUN: llc < %s -verify-machineinstrs -march=x86 | FileCheck %s

; CHECK: testl

; It's tempting to eliminate the testl instruction here and just use the
; EFLAGS value from the incl, however it can't be known whether the add
; will overflow, and if it does the incl would set OF, and the
; subsequent setg would return true.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i32 @f(i32 %j) nounwind readnone {
entry:
	%0 = add i32 %j, 1		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 0		; <i1> [#uses=1]
	%2 = zext i1 %1 to i32		; <i32> [#uses=1]
	ret i32 %2
}
