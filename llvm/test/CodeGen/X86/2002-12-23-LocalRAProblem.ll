; RUN: llvm-as < %s | llc -march=x86 -regalloc=simple

define i32 @main() {
	; %A = 0
        %A = add i32 0, 0		; <i32> [#uses=1]
        ; %B = 1
	%B = add i32 0, 1		; <i32> [#uses=2]
	br label %bb1
bb1:		; preds = %0
        ; %X = 0*1 = 0
 	%X = mul i32 %A, %B		; <i32> [#uses=0]
        ; %r = 0
	%R = sub i32 %B, 1		; <i32> [#uses=1]
	ret i32 %R
}
