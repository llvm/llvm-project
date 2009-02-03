; RUN: llvm-as < %s | opt -licm -disable-output
; PR2346
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-pc-linux-gnu"
	%struct._zval_struct = type { %union._double, i32, i8, i8, i8, i8 }
	%union._double = type { double }

define i8* @zend_fetch_resource(%struct._zval_struct** %passed_id, i32 %default_id, i8* %resource_type_name, i32* %found_resource_type, i32 %num_resource_types, ...) {
entry:
	br label %whilebody.i.i

whilebody.i.i:		; preds = %whilebody.i.i, %entry
	br i1 false, label %ifthen.i.i, label %whilebody.i.i

ifthen.i.i:		; preds = %whilebody.i.i
	br label %forcond

forcond:		; preds = %forbody, %ifthen.i.i
	br i1 false, label %forbody, label %afterfor

forbody:		; preds = %forcond
	va_arg i8** null, i32		; <i32>:0 [#uses=0]
	br i1 false, label %ifthen59, label %forcond

ifthen59:		; preds = %forbody
	unreachable

afterfor:		; preds = %forcond
	ret i8* null
}
