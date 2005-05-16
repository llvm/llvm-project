; This example was not getting level raised because of a negative index from
; this instruction:
;   %reg111 = add int %reg110, -12
; this testcase is distilled from this C source:
; void foo(int j, int *x) {
;   unsigned i;
;   for (i = 12; i < 14; ++i) 
;     x[j*i-12] = j;
; }

; RUN: llvm-as < %s | opt -raise | llvm-dis | grep ' cast ' | not grep '*'

implementation

void "foo"(int %j, int * %x)
begin
bb0:					;[#uses=0]
	br label %bb1

bb1:					;[#uses=2]
	%reg108 = cast int * %x to sbyte *		; <sbyte *> [#uses=1]
	%cond219 = setgt ulong 12, 13		; <bool> [#uses=1]
	br bool %cond219, label %bb3, label %bb2

bb2:					;[#uses=3]
	%cann-indvar = phi uint [ 0, %bb1 ], [ %add1-indvar, %bb2 ]		; <uint> [#uses=2]
	%reg117 = add uint %cann-indvar, 12		; <uint> [#uses=2]
	%add1-indvar = add uint %cann-indvar, 1		; <uint> [#uses=1]
	%cast224 = cast uint %reg117 to uint		; <uint> [#uses=1]
	%cast225 = cast uint %reg117 to int		; <int> [#uses=1]
	%reg110 = mul int %j, %cast225		; <int> [#uses=1]
	%reg111 = add int %reg110, -12		; <int> [#uses=1]
	%cast222 = cast int %reg111 to uint		; <uint> [#uses=1]
	%reg113 = shl uint %cast222, ubyte 2		; <uint> [#uses=1]
	%cast114 = cast uint %reg113 to ulong		; <ulong> [#uses=1]
	%cast115 = cast ulong %cast114 to sbyte *		; <sbyte *> [#uses=1]
	%reg116 = add sbyte * %reg108, %cast115		; <sbyte *> [#uses=1]
	%cast223 = cast sbyte * %reg116 to int *		; <int *> [#uses=1]
	store int %j, int * %cast223
	%reg118 = add uint %cast224, 1		; <uint> [#uses=1]
	%cond220 = setle uint %reg118, 13		; <bool> [#uses=1]
	br bool %cond220, label %bb2, label %bb3

bb3:					;[#uses=2]
	ret void
end
