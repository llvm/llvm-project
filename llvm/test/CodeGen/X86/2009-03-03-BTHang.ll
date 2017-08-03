; RUN: llc < %s -march=x86
; rdar://6642541

 	%struct.HandleBlock = type { [30 x i32], [990 x i8*], %struct.HandleBlockTrailer }
 	%struct.HandleBlockTrailer = type { %struct.HandleBlock* }

define hidden zeroext i8 @IsHandleAllocatedFromPool(i8** %h) nounwind optsize {
entry:
	%0 = ptrtoint i8** %h to i32		; <i32> [#uses=2]
	%1 = and i32 %0, -4096		; <i32> [#uses=1]
	%2 = inttoptr i32 %1 to %struct.HandleBlock*		; <%struct.HandleBlock*> [#uses=3]
	%3 = getelementptr %struct.HandleBlock, %struct.HandleBlock* %2, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%4 = load i32, i32* %3, align 4096		; <i32> [#uses=1]
	%5 = icmp eq i32 %4, 1751280747		; <i1> [#uses=1]
	br i1 %5, label %bb, label %bb1

bb:		; preds = %entry
	%6 = getelementptr %struct.HandleBlock, %struct.HandleBlock* %2, i32 0, i32 1		; <[990 x i8*]*> [#uses=1]
	%7 = ptrtoint [990 x i8*]* %6 to i32		; <i32> [#uses=1]
	%8 = sub i32 %0, %7		; <i32> [#uses=2]
	%9 = lshr i32 %8, 2		; <i32> [#uses=1]
	%10 = ashr i32 %8, 7		; <i32> [#uses=1]
	%11 = and i32 %10, 134217727		; <i32> [#uses=1]
	%12 = getelementptr %struct.HandleBlock, %struct.HandleBlock* %2, i32 0, i32 0, i32 %11		; <i32*> [#uses=1]
	%not.i = and i32 %9, 31		; <i32> [#uses=1]
	%13 = xor i32 %not.i, 31		; <i32> [#uses=1]
	%14 = shl i32 1, %13		; <i32> [#uses=1]
	%15 = load i32, i32* %12, align 4		; <i32> [#uses=1]
	%16 = and i32 %15, %14		; <i32> [#uses=1]
	%17 = icmp eq i32 %16, 0		; <i1> [#uses=1]
	%tmp = zext i1 %17 to i8		; <i8> [#uses=1]
	ret i8 %tmp

bb1:		; preds = %entry
	ret i8 0
}

