; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -regalloc=fast -optimize-regalloc=0 | FileCheck %s
; The first argument of subfc must not be the same as any other register.

; CHECK: APP
; CHECK: subc [[REG:[0-9]+]],
; CHECK-NOT: [[REG]]
; CHECK: NO_APP
; PR1357

;target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-unknown-linux-gnu"

;long long test(int A, int B, int C) {
;  unsigned X, Y;
;  __asm__ ("subf%I3c %1,%4,%3\n\tsubfze %0,%2"
;                 : "=r" (X), "=&r" (Y)
;                 : "r" (A), "rI" (B), "r" (C));
;  return ((long long)Y << 32) | X;
;}

define i64 @test(i32 %A, i32 %B, i32 %C) nounwind {
entry:
	%Y = alloca i32, align 4		; <ptr> [#uses=2]
	%tmp4 = call i32 asm "subf${3:I}c $1,$4,$3\0A\09subfze $0,$2", "=r,=*&r,r,rI,r"( ptr elementtype( i32) %Y, i32 %A, i32 %B, i32 %C )		; <i32> [#uses=1]
	%tmp5 = load i32, ptr %Y		; <i32> [#uses=1]
	%tmp56 = zext i32 %tmp5 to i64		; <i64> [#uses=1]
	%tmp7 = shl i64 %tmp56, 32		; <i64> [#uses=1]
	%tmp89 = zext i32 %tmp4 to i64		; <i64> [#uses=1]
	%tmp10 = or i64 %tmp7, %tmp89		; <i64> [#uses=1]
	ret i64 %tmp10
}
