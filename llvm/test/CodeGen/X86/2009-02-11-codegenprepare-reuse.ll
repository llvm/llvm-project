; RUN: llc < %s
; PR3537
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
	%struct.GetBitContext = type <{ ptr, ptr, i32, i32 }>

define i32 @alac_decode_frame() nounwind {
entry:
	%tmp2 = load ptr, ptr null		; <ptr> [#uses=2]
	%tmp34 = getelementptr i8, ptr %tmp2, i32 4		; <ptr> [#uses=2]
	%tmp15.i = getelementptr i8, ptr %tmp2, i32 12		; <ptr> [#uses=1]
	br i1 false, label %if.then43, label %if.end47

if.then43:		; preds = %entry
	ret i32 0

if.end47:		; preds = %entry
	%tmp5.i590 = load ptr, ptr %tmp34		; <ptr> [#uses=0]
	store i32 19, ptr %tmp15.i
	%tmp6.i569 = load ptr, ptr %tmp34		; <ptr> [#uses=0]
	%0 = call i32 asm "bswap   $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 0) nounwind		; <i32> [#uses=0]
	br i1 false, label %bb.nph, label %if.then63

if.then63:		; preds = %if.end47
	unreachable

bb.nph:		; preds = %if.end47
	%call9.i = call fastcc i32 @decode_scalar(ptr %tmp34, i32 0, i32 0, i32 0) nounwind		; <i32> [#uses=0]
	unreachable
}

declare fastcc i32 @decode_scalar(ptr nocapture, i32, i32, i32) nounwind
