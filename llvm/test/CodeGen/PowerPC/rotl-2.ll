; RUN: llc -verify-machineinstrs < %s -march=ppc32  | grep rotlwi | count 2
; RUN: llc -verify-machineinstrs < %s -march=ppc32  | grep clrlwi | count 2
; RUN: llc -verify-machineinstrs < %s -march=ppc32  | grep rotlw | count 4
; RUN: llc -verify-machineinstrs < %s -march=ppc32  | not grep or

define i32 @rotl32(i32 %A, i8 %Amt) nounwind {
	%shift.upgrd.1 = zext i8 %Amt to i32		; <i32> [#uses=1]
	%B = shl i32 %A, %shift.upgrd.1		; <i32> [#uses=1]
	%Amt2 = sub i8 32, %Amt		; <i8> [#uses=1]
	%shift.upgrd.2 = zext i8 %Amt2 to i32		; <i32> [#uses=1]
	%C = lshr i32 %A, %shift.upgrd.2		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @rotr32(i32 %A, i8 %Amt) nounwind {
	%shift.upgrd.3 = zext i8 %Amt to i32		; <i32> [#uses=1]
	%B = lshr i32 %A, %shift.upgrd.3		; <i32> [#uses=1]
	%Amt2 = sub i8 32, %Amt		; <i8> [#uses=1]
	%shift.upgrd.4 = zext i8 %Amt2 to i32		; <i32> [#uses=1]
	%C = shl i32 %A, %shift.upgrd.4		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @rotli32(i32 %A) nounwind {
	%B = shl i32 %A, 5		; <i32> [#uses=1]
	%C = lshr i32 %A, 27		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @rotri32(i32 %A) nounwind {
	%B = lshr i32 %A, 5		; <i32> [#uses=1]
	%C = shl i32 %A, 27		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

