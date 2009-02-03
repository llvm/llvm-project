; RUN: llvm-as < %s | llc -march=x86 | not grep rep
; RUN: llvm-as < %s | llc -march=x86 | grep memset

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind

define fastcc i32 @cli_scanzip(i32 %desc) nounwind {
entry:
	br label %bb8.i.i.i.i

bb8.i.i.i.i:		; preds = %bb8.i.i.i.i, %entry
	icmp eq i32 0, 0		; <i1>:0 [#uses=1]
	br i1 %0, label %bb61.i.i.i, label %bb8.i.i.i.i

bb32.i.i.i:		; preds = %bb61.i.i.i
	ptrtoint i8* %tail.0.i.i.i to i32		; <i32>:1 [#uses=1]
	sub i32 0, %1		; <i32>:2 [#uses=1]
	icmp sgt i32 %2, 19		; <i1>:3 [#uses=1]
	br i1 %3, label %bb34.i.i.i, label %bb61.i.i.i

bb34.i.i.i:		; preds = %bb32.i.i.i
	load i32* null, align 4		; <i32>:4 [#uses=1]
	icmp eq i32 %4, 101010256		; <i1>:5 [#uses=1]
	br i1 %5, label %bb8.i11.i.i.i, label %bb61.i.i.i

bb8.i11.i.i.i:		; preds = %bb8.i11.i.i.i, %bb34.i.i.i
	icmp eq i32 0, 0		; <i1>:6 [#uses=1]
	br i1 %6, label %cli_dbgmsg.exit49.i, label %bb8.i11.i.i.i

cli_dbgmsg.exit49.i:		; preds = %bb8.i11.i.i.i
	icmp eq [32768 x i8]* null, null		; <i1>:7 [#uses=1]
	br i1 %7, label %bb1.i28.i, label %bb8.i.i

bb61.i.i.i:		; preds = %bb61.i.i.i, %bb34.i.i.i, %bb32.i.i.i, %bb8.i.i.i.i
	%tail.0.i.i.i = getelementptr [1024 x i8]* null, i32 0, i32 0		; <i8*> [#uses=2]
	load i8* %tail.0.i.i.i, align 1		; <i8>:8 [#uses=1]
	icmp eq i8 %8, 80		; <i1>:9 [#uses=1]
	br i1 %9, label %bb32.i.i.i, label %bb61.i.i.i

bb1.i28.i:		; preds = %cli_dbgmsg.exit49.i
	call void @llvm.memset.i32( i8* null, i8 0, i32 88, i32 1 ) nounwind
	unreachable

bb8.i.i:		; preds = %bb8.i.i, %cli_dbgmsg.exit49.i
	br label %bb8.i.i
}
