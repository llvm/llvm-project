; RUN: llc -march=x86-64 < %s | FileCheck %s
; PR3701

define i64 @t(i64* %arg) nounwind {
	br i1 true, label %1, label %5

; <label>:1		; preds = %0
	%2 = icmp eq i64* null, %arg		; <i1> [#uses=1]
	%3 = tail call i64* asm sideeffect "movl %fs:0,$0", "=r,~{dirflag},~{fpsr},~{flags}"() nounwind		; <%struct.thread*> [#uses=0]
; CHECK: test
; CHECK-NEXT: j
	br i1 %2, label %4, label %5

; <label>:4		; preds = %1
	ret i64 1

; <label>:5		; preds = %1
	ret i64 0
}
