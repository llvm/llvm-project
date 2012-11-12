; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep "ro[rl]" | count 12

define i32 @rotl32(i32 %A, i8 %Amt) {
	%shift.upgrd.1 = zext i8 %Amt to i32		; <i32> [#uses=1]
	%B = shl i32 %A, %shift.upgrd.1		; <i32> [#uses=1]
	%Amt2 = sub i8 32, %Amt		; <i8> [#uses=1]
	%shift.upgrd.2 = zext i8 %Amt2 to i32		; <i32> [#uses=1]
	%C = lshr i32 %A, %shift.upgrd.2		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @rotr32(i32 %A, i8 %Amt) {
	%shift.upgrd.3 = zext i8 %Amt to i32		; <i32> [#uses=1]
	%B = lshr i32 %A, %shift.upgrd.3		; <i32> [#uses=1]
	%Amt2 = sub i8 32, %Amt		; <i8> [#uses=1]
	%shift.upgrd.4 = zext i8 %Amt2 to i32		; <i32> [#uses=1]
	%C = shl i32 %A, %shift.upgrd.4		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @rotli32(i32 %A) {
	%B = shl i32 %A, 5		; <i32> [#uses=1]
	%C = lshr i32 %A, 27		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @rotri32(i32 %A) {
	%B = lshr i32 %A, 5		; <i32> [#uses=1]
	%C = shl i32 %A, 27		; <i32> [#uses=1]
	%D = or i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i16 @rotl16(i16 %A, i8 %Amt) {
	%shift.upgrd.5 = zext i8 %Amt to i16		; <i16> [#uses=1]
	%B = shl i16 %A, %shift.upgrd.5		; <i16> [#uses=1]
	%Amt2 = sub i8 16, %Amt		; <i8> [#uses=1]
	%shift.upgrd.6 = zext i8 %Amt2 to i16		; <i16> [#uses=1]
	%C = lshr i16 %A, %shift.upgrd.6		; <i16> [#uses=1]
	%D = or i16 %B, %C		; <i16> [#uses=1]
	ret i16 %D
}

define i16 @rotr16(i16 %A, i8 %Amt) {
	%shift.upgrd.7 = zext i8 %Amt to i16		; <i16> [#uses=1]
	%B = lshr i16 %A, %shift.upgrd.7		; <i16> [#uses=1]
	%Amt2 = sub i8 16, %Amt		; <i8> [#uses=1]
	%shift.upgrd.8 = zext i8 %Amt2 to i16		; <i16> [#uses=1]
	%C = shl i16 %A, %shift.upgrd.8		; <i16> [#uses=1]
	%D = or i16 %B, %C		; <i16> [#uses=1]
	ret i16 %D
}

define i16 @rotli16(i16 %A) {
	%B = shl i16 %A, 5		; <i16> [#uses=1]
	%C = lshr i16 %A, 11		; <i16> [#uses=1]
	%D = or i16 %B, %C		; <i16> [#uses=1]
	ret i16 %D
}

define i16 @rotri16(i16 %A) {
	%B = lshr i16 %A, 5		; <i16> [#uses=1]
	%C = shl i16 %A, 11		; <i16> [#uses=1]
	%D = or i16 %B, %C		; <i16> [#uses=1]
	ret i16 %D
}

define i8 @rotl8(i8 %A, i8 %Amt) {
	%B = shl i8 %A, %Amt		; <i8> [#uses=1]
	%Amt2 = sub i8 8, %Amt		; <i8> [#uses=1]
	%C = lshr i8 %A, %Amt2		; <i8> [#uses=1]
	%D = or i8 %B, %C		; <i8> [#uses=1]
	ret i8 %D
}

define i8 @rotr8(i8 %A, i8 %Amt) {
	%B = lshr i8 %A, %Amt		; <i8> [#uses=1]
	%Amt2 = sub i8 8, %Amt		; <i8> [#uses=1]
	%C = shl i8 %A, %Amt2		; <i8> [#uses=1]
	%D = or i8 %B, %C		; <i8> [#uses=1]
	ret i8 %D
}

define i8 @rotli8(i8 %A) {
	%B = shl i8 %A, 5		; <i8> [#uses=1]
	%C = lshr i8 %A, 3		; <i8> [#uses=1]
	%D = or i8 %B, %C		; <i8> [#uses=1]
	ret i8 %D
}

define i8 @rotri8(i8 %A) {
	%B = lshr i8 %A, 5		; <i8> [#uses=1]
	%C = shl i8 %A, 3		; <i8> [#uses=1]
	%D = or i8 %B, %C		; <i8> [#uses=1]
	ret i8 %D
}
