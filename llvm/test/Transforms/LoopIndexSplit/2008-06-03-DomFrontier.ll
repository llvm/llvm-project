; RUN: llvm-as < %s | opt -loop-rotate -loop-unswitch -loop-index-split -instcombine -disable-output
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.__CFData = type opaque
	%struct.__CFString = type opaque

define %struct.__CFData* @WirelessCreatePSK(%struct.__CFString* %inPassphrase, %struct.__CFData* %inSSID) nounwind  {
entry:
	br label %bb52

bb52:		; preds = %bb142, %bb52, %entry
	br i1 false, label %bb142, label %bb52

bb63:		; preds = %bb142, %bb131
	%t.0.reg2mem.0 = phi i32 [ %tmp133, %bb131 ], [ 0, %bb142 ]		; <i32> [#uses=2]
	%tmp65 = icmp ult i32 %t.0.reg2mem.0, 16		; <i1> [#uses=1]
	br i1 %tmp65, label %bb68, label %bb89

bb68:		; preds = %bb63
	br label %bb131

bb89:		; preds = %bb63
	br label %bb131

bb131:		; preds = %bb89, %bb68
	%tmp133 = add i32 %t.0.reg2mem.0, 1		; <i32> [#uses=2]
	%tmp136 = icmp ult i32 %tmp133, 80		; <i1> [#uses=1]
	br i1 %tmp136, label %bb63, label %bb142

bb142:		; preds = %bb131, %bb52
	br i1 undef, label %bb63, label %bb52
}
