; RUN: llvm-as < %s | opt -loop-index-split -disable-output -stats |& \
; RUN: not grep "loop-index-split" 

; Induction variable decrement is not yet handled.
; pr1912.bc
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"
	%struct.cset = type { i8*, i8, i8, i32, i8* }
	%struct.parse = type { i8*, i8*, i32, i32*, i32, i32, i32, %struct.re_guts*, [10 x i32], [10 x i32] }
	%struct.re_guts = type { i32, i32*, i32, i32, %struct.cset*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i32, i32, i32, i32, [1 x i8] }

define fastcc void @p_bracket(%struct.parse* %p) {
entry:
	br i1 false, label %bb160, label %bb195

bb160:		; preds = %entry
	br i1 false, label %bb.i169, label %bb9.i

bb195:		; preds = %entry
	ret void

bb.i169:		; preds = %bb160
	br i1 false, label %bb372, label %bb565

bb9.i:		; preds = %bb160
	ret void

bb372:		; preds = %bb418, %bb.i169
	%i1.0.reg2mem.0 = phi i32 [ %i1.0, %bb418 ], [ 0, %bb.i169 ]		; <i32> [#uses=2]
	%tmp3.i.i.i170 = icmp ult i32 %i1.0.reg2mem.0, 128		; <i1> [#uses=1]
	br i1 %tmp3.i.i.i170, label %bb.i.i173, label %bb13.i.i

bb.i.i173:		; preds = %bb372
	br label %bb418

bb13.i.i:		; preds = %bb372
	br label %bb418

bb418:		; preds = %bb13.i.i, %bb.i.i173
	%i1.0 = add i32 %i1.0.reg2mem.0, -1		; <i32> [#uses=2]
	%tmp420 = icmp sgt i32 %i1.0, -1		; <i1> [#uses=1]
	br i1 %tmp420, label %bb372, label %bb565

bb565:		; preds = %bb418, %bb.i169
	ret void
}
