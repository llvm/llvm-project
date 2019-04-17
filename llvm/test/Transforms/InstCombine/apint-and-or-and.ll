; If we have an 'and' of the result of an 'or', and one of the 'or' operands
; cannot have contributed any of the resultant bits, delete the or.  This
; occurs for very common C/C++ code like this:
;
; struct foo { int A : 16; int B : 16; };
; void test(struct foo *F, int X, int Y) {
;        F->A = X; F->B = Y;
; }
;
; Which corresponds to test1.
; 
; This tests arbitrary precision integers.

; RUN: opt < %s -instcombine -S | not grep "or "
; END.

define i17 @test1(i17 %X, i17 %Y) {
	%A = and i17 %X, 7
	%B = and i17 %Y, 8
	%C = or i17 %A, %B
	%D = and i17 %C, 7  ;; This cannot include any bits from %Y!
	ret i17 %D
}

define i49 @test3(i49 %X, i49 %Y) {
	%B = shl i49 %Y, 1
	%C = or i49 %X, %B
	%D = and i49 %C, 1  ;; This cannot include any bits from %Y!
	ret i49 %D
}

define i67 @test4(i67 %X, i67 %Y) {
	%B = lshr i67 %Y, 66
	%C = or i67 %X, %B
	%D = and i67 %C, 2  ;; This cannot include any bits from %Y!
	ret i67 %D
}

define i231 @or_test1(i231 %X, i231 %Y) {
	%A = and i231 %X, 1
	%B = or i231 %A, 1     ;; This cannot include any bits from X!
	ret i231 %B
}

define i7 @or_test2(i7 %X, i7 %Y) {
	%A = shl i7 %X, 6
	%B = or i7 %A, 64     ;; This cannot include any bits from X!
	ret i7 %B
}

