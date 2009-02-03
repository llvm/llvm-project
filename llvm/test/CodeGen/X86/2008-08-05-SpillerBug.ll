; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -disable-fp-elim -stats -info-output-file - | not grep {Number of dead stores elided}
; PR2568

@g_3 = external global i16		; <i16*> [#uses=1]
@g_5 = external global i32		; <i32*> [#uses=3]

declare i32 @func_15(i16 signext , i16 signext , i32) nounwind 

define void @func_9_entry_2E_ce(i8 %p_11) nounwind {
newFuncRoot:
	br label %entry.ce

entry.ce.ret.exitStub:		; preds = %entry.ce
	ret void

entry.ce:		; preds = %newFuncRoot
	load i16* @g_3, align 2		; <i16>:0 [#uses=1]
	icmp sgt i16 %0, 0		; <i1>:1 [#uses=1]
	zext i1 %1 to i32		; <i32>:2 [#uses=1]
	load i32* @g_5, align 4		; <i32>:3 [#uses=4]
	icmp ugt i32 %2, %3		; <i1>:4 [#uses=1]
	zext i1 %4 to i32		; <i32>:5 [#uses=1]
	icmp eq i32 %3, 0		; <i1>:6 [#uses=1]
	%.0 = select i1 %6, i32 1, i32 %3		; <i32> [#uses=1]
	urem i32 1, %.0		; <i32>:7 [#uses=2]
	sext i8 %p_11 to i16		; <i16>:8 [#uses=1]
	trunc i32 %3 to i16		; <i16>:9 [#uses=1]
	tail call i32 @func_15( i16 signext  %8, i16 signext  %9, i32 1 ) nounwind 		; <i32>:10 [#uses=0]
	load i32* @g_5, align 4		; <i32>:11 [#uses=1]
	trunc i32 %11 to i16		; <i16>:12 [#uses=1]
	tail call i32 @func_15( i16 signext  %12, i16 signext  1, i32 %7 ) nounwind 		; <i32>:13 [#uses=0]
	sext i8 %p_11 to i32		; <i32>:14 [#uses=1]
	%p_11.lobit = lshr i8 %p_11, 7		; <i8> [#uses=1]
	%tmp = zext i8 %p_11.lobit to i32		; <i32> [#uses=1]
	%tmp.not = xor i32 %tmp, 1		; <i32> [#uses=1]
	%.015 = ashr i32 %14, %tmp.not		; <i32> [#uses=2]
	icmp eq i32 %.015, 0		; <i1>:15 [#uses=1]
	%.016 = select i1 %15, i32 1, i32 %.015		; <i32> [#uses=1]
	udiv i32 %7, %.016		; <i32>:16 [#uses=1]
	icmp ult i32 %5, %16		; <i1>:17 [#uses=1]
	zext i1 %17 to i32		; <i32>:18 [#uses=1]
	store i32 %18, i32* @g_5, align 4
	br label %entry.ce.ret.exitStub
}
