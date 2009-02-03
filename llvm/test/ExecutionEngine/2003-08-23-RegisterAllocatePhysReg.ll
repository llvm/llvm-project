; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

; This testcase exposes a bug in the local register allocator where it runs out
; of registers (due to too many overlapping live ranges), but then attempts to
; use the ESP register (which is not allocatable) to hold a value.

define i32 @main(i32 %A) {
        ; ESP gets used again...
	%Ap2 = alloca i32, i32 %A		; <i32*> [#uses=11]
	; Produce lots of overlapping live ranges
        %B = add i32 %A, 1		; <i32> [#uses=1]
	%C = add i32 %A, 2		; <i32> [#uses=1]
	%D = add i32 %A, 3		; <i32> [#uses=1]
	%E = add i32 %A, 4		; <i32> [#uses=1]
	%F = add i32 %A, 5		; <i32> [#uses=1]
	%G = add i32 %A, 6		; <i32> [#uses=1]
	%H = add i32 %A, 7		; <i32> [#uses=1]
	%I = add i32 %A, 8		; <i32> [#uses=1]
	%J = add i32 %A, 9		; <i32> [#uses=1]
	%K = add i32 %A, 10		; <i32> [#uses=1]
        ; Uses of all of the values
	store i32 %A, i32* %Ap2
	store i32 %B, i32* %Ap2
	store i32 %C, i32* %Ap2
	store i32 %D, i32* %Ap2
	store i32 %E, i32* %Ap2
	store i32 %F, i32* %Ap2
	store i32 %G, i32* %Ap2
	store i32 %H, i32* %Ap2
	store i32 %I, i32* %Ap2
	store i32 %J, i32* %Ap2
	store i32 %K, i32* %Ap2
	ret i32 0
}
