; RUN: llc < %s -march=x86-64 | FileCheck %s

define i64 @t(i64 %a, i64 %b) nounwind ssp {
entry:
; CHECK-LABEL: t:
	%asmtmp = tail call i64 asm "rorq $1,$0", "=r,J,0,~{dirflag},~{fpsr},~{flags},~{cc}"(i32 1, i64 %a) nounwind		; <i64> [#uses=1]
; CHECK:      #APP
; CHECK-NEXT: rorq    %[[REG1:.*]]
; CHECK-NEXT: #NO_APP
	%asmtmp1 = tail call i64 asm "rorq $1,$0", "=r,J,0,~{dirflag},~{fpsr},~{flags},~{cc}"(i32 1, i64 %b) nounwind		; <i64> [#uses=1]
; CHECK-NEXT: #APP
; CHECK-NEXT: rorq    %[[REG2:.*]]
; CHECK-NEXT: #NO_APP
	%0 = add i64 %asmtmp1, %asmtmp		; <i64> [#uses=1]
; CHECK-NEXT: leaq    (%[[REG2]],%[[REG1]]), %rax
	ret i64 %0
; CHECK:      retq
}
