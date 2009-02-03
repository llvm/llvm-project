; RUN: llvm-as < %s | llc -march=alpha

;FIXME: this should produce no mul inst.  But not crashing will have to do for now

; ModuleID = 'Output/bugpoint-train/bugpoint-reduced-simplified.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
target triple = "alphaev6-unknown-linux-gnu"

define fastcc i32 @getcount(i32 %s) {
cond_next43:		; preds = %bb27
	%tmp431 = mul i32 %s, -3
	ret i32 %tmp431
}
