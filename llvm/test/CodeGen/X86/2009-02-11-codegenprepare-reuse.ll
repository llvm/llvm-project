; RUN: llc < %s
; PR3537
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
	%struct.GetBitContext = type <{ i8*, i8*, i32, i32 }>

define i32 @alac_decode_frame() nounwind {
entry:
	%tmp2 = load i8** null		; <i8*> [#uses=2]
	%tmp34 = getelementptr i8* %tmp2, i32 4		; <i8*> [#uses=2]
	%tmp5.i424 = bitcast i8* %tmp34 to i8**		; <i8**> [#uses=2]
	%tmp15.i = getelementptr i8* %tmp2, i32 12		; <i8*> [#uses=1]
	%0 = bitcast i8* %tmp15.i to i32*		; <i32*> [#uses=1]
	br i1 false, label %if.then43, label %if.end47

if.then43:		; preds = %entry
	ret i32 0

if.end47:		; preds = %entry
	%tmp5.i590 = load i8** %tmp5.i424		; <i8*> [#uses=0]
	store i32 19, i32* %0
	%tmp6.i569 = load i8** %tmp5.i424		; <i8*> [#uses=0]
	%1 = call i32 asm "bswap   $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind		; <i32> [#uses=0]
	br i1 false, label %bb.nph, label %if.then63

if.then63:		; preds = %if.end47
	unreachable

bb.nph:		; preds = %if.end47
	%2 = bitcast i8* %tmp34 to %struct.GetBitContext*		; <%struct.GetBitContext*> [#uses=1]
	%call9.i = call fastcc i32 @decode_scalar(%struct.GetBitContext* %2, i32 0, i32 0, i32 0) nounwind		; <i32> [#uses=0]
	unreachable
}

declare fastcc i32 @decode_scalar(%struct.GetBitContext* nocapture, i32, i32, i32) nounwind
