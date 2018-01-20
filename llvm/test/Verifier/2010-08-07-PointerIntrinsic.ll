; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: assembly parsed, but does not verify as correct
; PR7316

; XFAIL: *
; The test case is buggy, it supposed to check that we reject memcpy with vector
; pointer arguments. Now we don't reject such memcpy and the test case would
; fail if it was correct. Because it used the wrong signature for memcpy (the
; last isVolatile argument was missing) it was rejected by the verifier and
; didn't fail. Fix the memcpy signature and mark it as an expected failure
; for now.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"
target triple = "x86-unknown-unknown"
@aa = global [32 x i8] zeroinitializer, align 1
@bb = global [16 x i8] zeroinitializer, align 1
define void @x() nounwind {
L.0:
	%0 = getelementptr [32 x i8], [32 x i8]* @aa, i32 0, i32 4
	%1 = bitcast i8* %0 to [16 x i8]*
	%2 = bitcast [16 x i8]* %1 to [0 x i8]*
	%3 = getelementptr [16 x i8], [16 x i8]* @bb
	%4 = bitcast [16 x i8]* %3 to [0 x i8]*
	call void @llvm.memcpy.p0a0i8.p0a0i8.i32([0 x i8]* %2, [0 x i8]* %4, i32 16, i32 1, i1 false)
	br label %return
return:
	ret void
}
declare void @llvm.memcpy.p0a0i8.p0a0i8.i32([0 x i8]* nocapture, [0 x i8]* nocapture readonly, i32, i1) nounwind
