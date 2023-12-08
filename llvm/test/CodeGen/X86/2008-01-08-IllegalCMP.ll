; RUN: llc < %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

define i64 @__absvdi2(i64 %a) nounwind  {
entry:
	%w.0 = select i1 false, i64 0, i64 %a		; <i64> [#uses=2]
	%tmp9 = icmp slt i64 %w.0, 0		; <i1> [#uses=1]
	br i1 %tmp9, label %bb12, label %bb13

bb12:		; preds = %entry
	unreachable

bb13:		; preds = %entry
	ret i64 %w.0
}
