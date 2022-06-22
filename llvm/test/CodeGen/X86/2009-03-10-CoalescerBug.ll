; RUN: llc < %s -mtriple=x86_64-apple-darwin
; rdar://r6661945

	%struct.WINDOW = type { i16, i16, i16, i16, i16, i16, i16, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, ptr, i16, i16, i32, i32, ptr, %struct.pdat, i16, %struct.cchar_t }
	%struct.cchar_t = type { i32, [5 x i32] }
	%struct.ldat = type { ptr, i16, i16, i16 }
	%struct.pdat = type { i16, i16, i16, i16, i16, i16 }

define i32 @pnoutrefresh(ptr %win, i32 %pminrow, i32 %pmincol, i32 %sminrow, i32 %smincol, i32 %smaxrow, i32 %smaxcol) nounwind optsize ssp {
entry:
	%0 = load i16, ptr null, align 4		; <i16> [#uses=2]
	%1 = icmp sgt i16 0, %0		; <i1> [#uses=1]
	br i1 %1, label %bb12, label %bb13

bb12:		; preds = %entry
	%2 = sext i16 %0 to i32		; <i32> [#uses=1]
	%3 = sub i32 %2, 0		; <i32> [#uses=1]
	%4 = add i32 %3, %smaxrow		; <i32> [#uses=2]
	%5 = trunc i32 %4 to i16		; <i16> [#uses=1]
	%6 = add i16 0, %5		; <i16> [#uses=1]
	br label %bb13

bb13:		; preds = %bb12, %entry
	%pmaxrow.0 = phi i16 [ %6, %bb12 ], [ 0, %entry ]		; <i16> [#uses=0]
	%smaxrow_addr.0 = phi i32 [ %4, %bb12 ], [ %smaxrow, %entry ]		; <i32> [#uses=1]
	%7 = trunc i32 %smaxrow_addr.0 to i16		; <i16> [#uses=0]
	ret i32 0
}
