; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -disable-fp-elim
; XFAIL: *
; Expected to run out of registers during allocation.
; rdar://6251720

	%struct.CABACContext = type { i32, i32, i8* }
	%struct.H264Context = type { %struct.CABACContext, [460 x i8] }
@coeff_abs_level_m1_offset = common global [6 x i32] zeroinitializer		; <[6 x i32]*> [#uses=1]
@coeff_abs_level1_ctx = common global [8 x i8] zeroinitializer		; <[8 x i8]*> [#uses=1]

define i32 @decode_cabac_residual(%struct.H264Context* %h, i32 %cat) nounwind {
entry:
	%0 = getelementptr [6 x i32]* @coeff_abs_level_m1_offset, i32 0, i32 %cat		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = load i8* getelementptr ([8 x i8]* @coeff_abs_level1_ctx, i32 0, i32 0), align 1		; <i8> [#uses=1]
	%3 = zext i8 %2 to i32		; <i32> [#uses=1]
	%.sum = add i32 %3, %1		; <i32> [#uses=1]
	%4 = getelementptr %struct.H264Context* %h, i32 0, i32 1, i32 %.sum		; <i8*> [#uses=2]
	%5 = getelementptr %struct.H264Context* %h, i32 0, i32 0, i32 0		; <i32*> [#uses=2]
	%6 = getelementptr %struct.H264Context* %h, i32 0, i32 0, i32 1		; <i32*> [#uses=2]
	%7 = getelementptr %struct.H264Context* %h, i32 0, i32 0, i32 2		; <i8**> [#uses=2]
	%8 = load i32* %5, align 4		; <i32> [#uses=1]
	%9 = load i32* %6, align 4		; <i32> [#uses=1]
	%10 = load i8* %4, align 4		; <i8> [#uses=1]
	%asmtmp = tail call { i32, i32, i32, i32 } asm sideeffect "#$0 $1 $2 $3 $4 $5", "=&{di},=r,=r,=*m,=&q,=*imr,1,2,*m,5,~{dirflag},~{fpsr},~{flags},~{cx}"(i8** %7, i8* %4, i32 %8, i32 %9, i8** %7, i8 %10) nounwind		; <{ i32, i32, i32, i32 }> [#uses=3]
	%asmresult = extractvalue { i32, i32, i32, i32 } %asmtmp, 0		; <i32> [#uses=1]
	%asmresult1 = extractvalue { i32, i32, i32, i32 } %asmtmp, 1		; <i32> [#uses=1]
	store i32 %asmresult1, i32* %5
	%asmresult2 = extractvalue { i32, i32, i32, i32 } %asmtmp, 2		; <i32> [#uses=1]
	store i32 %asmresult2, i32* %6
	ret i32 %asmresult
}
