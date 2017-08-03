; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | FileCheck %s

; CHECK: inc
; CHECK-NOT: PTR
; CHECK: {{$}}

define signext   i16 @t(i32* %bitptr, i32* %source, i8** %byteptr, i32 %scale, i32 %round) {
entry:
	br label %bb

bb:		; preds = %cond_next391, %entry
	%cnt.0 = phi i32 [ 0, %entry ], [ %tmp422445, %cond_next391 ]		; <i32> [#uses=1]
	%v.1 = phi i32 [ undef, %entry ], [ %tmp411, %cond_next391 ]		; <i32> [#uses=0]
	br i1 false, label %cond_true, label %cond_next127

cond_true:		; preds = %bb
	store i8* null, i8** %byteptr, align 4
	store i8* null, i8** %byteptr, align 4
	br label %cond_next127

cond_next127:		; preds = %cond_true, %bb
	%tmp151 = add i32 0, %round		; <i32> [#uses=1]
	%tmp153 = ashr i32 %tmp151, %scale		; <i32> [#uses=2]
	%tmp154155 = trunc i32 %tmp153 to i16		; <i16> [#uses=1]
	%tmp154155156 = sext i16 %tmp154155 to i32		; <i32> [#uses=1]
	%tmp158 = xor i32 %tmp154155156, %tmp153		; <i32> [#uses=1]
	%tmp160 = or i32 %tmp158, %cnt.0		; <i32> [#uses=1]
	%tmp171 = load i32, i32* %bitptr, align 4		; <i32> [#uses=1]
	%tmp180181 = sext i16 0 to i32		; <i32> [#uses=3]
	%tmp183 = add i32 %tmp160, 1		; <i32> [#uses=1]
	br i1 false, label %cond_true188, label %cond_next245

cond_true188:		; preds = %cond_next127
	ret i16 0

cond_next245:		; preds = %cond_next127
	%tmp249 = ashr i32 %tmp180181, 8		; <i32> [#uses=1]
	%tmp250 = add i32 %tmp171, %tmp249		; <i32> [#uses=1]
	%tmp253444 = lshr i32 %tmp180181, 4		; <i32> [#uses=1]
	%tmp254 = and i32 %tmp253444, 15		; <i32> [#uses=1]
	%tmp256 = and i32 %tmp180181, 15		; <i32> [#uses=2]
	%tmp264 = icmp ugt i32 %tmp250, 15		; <i1> [#uses=1]
	br i1 %tmp264, label %cond_true267, label %cond_next391

cond_true267:		; preds = %cond_next245
	store i8* null, i8** %byteptr, align 4
	store i8* null, i8** %byteptr, align 4
	br i1 false, label %cond_true289, label %cond_next327

cond_true289:		; preds = %cond_true267
	ret i16 0

cond_next327:		; preds = %cond_true267
	br i1 false, label %cond_true343, label %cond_next385

cond_true343:		; preds = %cond_next327
	%tmp345 = load i8*, i8** %byteptr, align 4		; <i8*> [#uses=1]
	store i8* null, i8** %byteptr, align 4
	br i1 false, label %cond_next385, label %cond_true352

cond_true352:		; preds = %cond_true343
	store i8* %tmp345, i8** %byteptr, align 4
	br i1 false, label %cond_true364, label %cond_next385

cond_true364:		; preds = %cond_true352
	ret i16 0

cond_next385:		; preds = %cond_true352, %cond_true343, %cond_next327
	br label %cond_next391

cond_next391:		; preds = %cond_next385, %cond_next245
	%tmp393 = load i32, i32* %source, align 4		; <i32> [#uses=1]
	%tmp395 = load i32, i32* %bitptr, align 4		; <i32> [#uses=2]
	%tmp396 = shl i32 %tmp393, %tmp395		; <i32> [#uses=1]
	%tmp398 = sub i32 32, %tmp256		; <i32> [#uses=1]
	%tmp405 = lshr i32 %tmp396, 31		; <i32> [#uses=1]
	%tmp406 = add i32 %tmp405, -1		; <i32> [#uses=1]
	%tmp409 = lshr i32 %tmp406, %tmp398		; <i32> [#uses=1]
	%tmp411 = sub i32 0, %tmp409		; <i32> [#uses=1]
	%tmp422445 = add i32 %tmp254, %tmp183		; <i32> [#uses=2]
	%tmp426447 = add i32 %tmp395, %tmp256		; <i32> [#uses=1]
	store i32 %tmp426447, i32* %bitptr, align 4
	%tmp429448 = icmp ult i32 %tmp422445, 63		; <i1> [#uses=1]
	br i1 %tmp429448, label %bb, label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %cond_next391
	ret i16 0
}
