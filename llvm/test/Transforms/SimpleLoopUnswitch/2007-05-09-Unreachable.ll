; PR1333
; RUN: opt < %s -simple-loop-unswitch -disable-output
; RUN: opt < %s -simple-loop-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"
	%struct.ada__streams__root_stream_type = type { %struct.ada__tags__dispatch_table* }
	%struct.ada__tags__dispatch_table = type { [1 x i8*] }
	%struct.quotes__T173s = type { i8, %struct.quotes__T173s__T174s, [2 x [1 x double]], [2 x i16], i64, i8 }
	%struct.quotes__T173s__T174s = type { i8, i8, i8, i16, i16, [2 x [1 x double]] }

define void @quotes__write_quote() {
entry:
	%tmp606.i = icmp eq i32 0, 0		; <i1> [#uses=1]
	br label %bb
bb:		; preds = %cond_next73, %bb, %entry
	br i1 false, label %bb51, label %bb
bb51:		; preds = %cond_next73, %bb
	br i1 %tmp606.i, label %quotes__bid_ask_depth_offset_matrices__get_price.exit, label %cond_true.i
cond_true.i:		; preds = %bb51
	unreachable
quotes__bid_ask_depth_offset_matrices__get_price.exit:		; preds = %bb51
	br i1 false, label %cond_next73, label %cond_true72
cond_true72:		; preds = %quotes__bid_ask_depth_offset_matrices__get_price.exit
	unreachable
cond_next73:		; preds = %quotes__bid_ask_depth_offset_matrices__get_price.exit
	br i1 false, label %bb, label %bb51
}

