; RUN: llvm-as < %s | llc
; PR1011	
%struct.mng_data = type { i8* (%struct.mng_data*, i32)*, i32, i32, i32, i8, i8, i32, i32, i32, i32, i32 }

define void @mng_display_bgr565() {
entry:
	br i1 false, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	br i1 false, label %cond_true48, label %cond_next80

cond_true48:		; preds = %bb.preheader
	%tmp = load i8* null		; <i8> [#uses=1]
	%tmp51 = zext i8 %tmp to i16		; <i16> [#uses=1]
	%tmp99 = load i8* null		; <i8> [#uses=1]
	%tmp54 = bitcast i8 %tmp99 to i8		; <i8> [#uses=1]
	%tmp54.upgrd.1 = zext i8 %tmp54 to i32		; <i32> [#uses=1]
	%tmp55 = lshr i32 %tmp54.upgrd.1, 3		; <i32> [#uses=1]
	%tmp55.upgrd.2 = trunc i32 %tmp55 to i16		; <i16> [#uses=1]
	%tmp52 = shl i16 %tmp51, 5		; <i16> [#uses=1]
	%tmp56 = and i16 %tmp55.upgrd.2, 28		; <i16> [#uses=1]
	%tmp57 = or i16 %tmp56, %tmp52		; <i16> [#uses=1]
	%tmp60 = zext i16 %tmp57 to i32		; <i32> [#uses=1]
	%tmp62 = xor i32 0, 65535		; <i32> [#uses=1]
	%tmp63 = mul i32 %tmp60, %tmp62		; <i32> [#uses=1]
	%tmp65 = add i32 0, %tmp63		; <i32> [#uses=1]
	%tmp69 = add i32 0, %tmp65		; <i32> [#uses=1]
	%tmp70 = lshr i32 %tmp69, 16		; <i32> [#uses=1]
	%tmp70.upgrd.3 = trunc i32 %tmp70 to i16		; <i16> [#uses=1]
	%tmp75 = lshr i16 %tmp70.upgrd.3, 8		; <i16> [#uses=1]
	%tmp75.upgrd.4 = trunc i16 %tmp75 to i8		; <i8> [#uses=1]
	%tmp76 = lshr i8 %tmp75.upgrd.4, 5		; <i8> [#uses=1]
	store i8 %tmp76, i8* null
	ret void

cond_next80:		; preds = %bb.preheader
	ret void

return:		; preds = %entry
	ret void
}
