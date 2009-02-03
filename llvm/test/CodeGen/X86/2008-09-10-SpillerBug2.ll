; RUN: llvm-as < %s | llc -march=x86 | grep movw | not grep %e.x
; PR2681

@g_491 = external global i32		; <i32*> [#uses=1]
@g_897 = external global i16		; <i16*> [#uses=1]

define i32 @func_7(i16 signext %p_9) nounwind {
entry:
	%p_9.addr = alloca i16		; <i16*> [#uses=2]
	%l_1122 = alloca i16, align 2		; <i16*> [#uses=1]
	%l_1128 = alloca i32, align 4		; <i32*> [#uses=1]
	%l_1129 = alloca i32, align 4		; <i32*> [#uses=1]
	%l_1130 = alloca i32, align 4		; <i32*> [#uses=1]
	%tmp14 = load i16* %l_1122		; <i16> [#uses=1]
	%conv15 = sext i16 %tmp14 to i32		; <i32> [#uses=1]
	%tmp16 = load i16* %p_9.addr		; <i16> [#uses=1]
	%conv17 = sext i16 %tmp16 to i32		; <i32> [#uses=1]
	%xor = xor i32 %conv15, %conv17		; <i32> [#uses=1]
	%tmp18 = load i32* null		; <i32> [#uses=1]
	%or = or i32 %xor, %tmp18		; <i32> [#uses=1]
	%conv19 = trunc i32 %or to i16		; <i16> [#uses=1]
	%tmp28 = load i16* %p_9.addr		; <i16> [#uses=1]
	%tmp33 = load i16* @g_897		; <i16> [#uses=1]
	%tmp34 = load i32* @g_491		; <i32> [#uses=1]
	%conv35 = trunc i32 %tmp34 to i16		; <i16> [#uses=1]
	%tmp36 = load i16* null		; <i16> [#uses=1]
	%conv37 = trunc i16 %tmp36 to i8		; <i8> [#uses=1]
	%tmp38 = load i32* %l_1128		; <i32> [#uses=1]
	%conv39 = sext i32 %tmp38 to i64		; <i64> [#uses=1]
	%tmp42 = load i32* %l_1129		; <i32> [#uses=1]
	%conv43 = trunc i32 %tmp42 to i16		; <i16> [#uses=1]
	%tmp44 = load i32* %l_1130		; <i32> [#uses=1]
	%conv45 = sext i32 %tmp44 to i64		; <i64> [#uses=1]
	%call46 = call i32 @func_18( i16 zeroext 0, i16 zeroext 0, i16 zeroext %tmp33, i16 zeroext %conv35, i8 zeroext %conv37, i64 %conv39, i32 0, i16 zeroext %conv43, i64 %conv45, i8 zeroext 1 )		; <i32> [#uses=0]
	%call48 = call i32 @func_18( i16 zeroext 0, i16 zeroext 0, i16 zeroext 0, i16 zeroext 1, i8 zeroext 0, i64 0, i32 1, i16 zeroext %tmp28, i64 0, i8 zeroext 1 )		; <i32> [#uses=0]
	%call50 = call i32 @func_18( i16 zeroext 1, i16 zeroext 0, i16 zeroext 0, i16 zeroext 1, i8 zeroext 0, i64 0, i32 1, i16 zeroext %conv19, i64 0, i8 zeroext 1 )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @func_18(i16 zeroext, i16 zeroext, i16 zeroext, i16 zeroext, i8 zeroext, i64, i32, i16 zeroext, i64, i8 zeroext)
