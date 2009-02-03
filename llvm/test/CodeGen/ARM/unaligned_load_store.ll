; RUN: llvm-as < %s | \
; RUN:   llc -march=arm -o %t -f
; RUN: grep ldrb %t | count 4
; RUN: grep strb %t | count 4


	%struct.p = type <{ i8, i32 }>
@t = global %struct.p <{ i8 1, i32 10 }>		; <%struct.p*> [#uses=1]
@u = weak global %struct.p zeroinitializer		; <%struct.p*> [#uses=1]

define i32 @main() {
entry:
	%tmp3 = load i32* getelementptr (%struct.p* @t, i32 0, i32 1), align 1		; <i32> [#uses=2]
	store i32 %tmp3, i32* getelementptr (%struct.p* @u, i32 0, i32 1), align 1
	ret i32 %tmp3
}
