; RUN: llvm-as < %s | opt -loop-index-split -disable-output 
; PR1828.bc
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%RPyOpaque_RuntimeTypeInfo = type opaque*
	%arraytype_Char_1 = type { i32, [0 x i8] }
	%arraytype_Signed = type { i32, [0 x i32] }
	%functiontype_11 = type %structtype_object* ()
	%functiontype_360 = type %structtype_rpy_string* (%structtype_pypy.rlib.rbigint.rbigint*, %structtype_rpy_string*, %structtype_rpy_string*, %structtype_rpy_string*)
	%structtype_list_18 = type { i32, %arraytype_Signed* }
	%structtype_object = type { %structtype_object_vtable* }
	%structtype_object_vtable = type { i32, i32, %RPyOpaque_RuntimeTypeInfo*, %arraytype_Char_1*, %functiontype_11* }
	%structtype_pypy.rlib.rbigint.rbigint = type { %structtype_object, %structtype_list_18*, i32 }
	%structtype_rpy_string = type { i32, %arraytype_Char_1 }

define fastcc %structtype_rpy_string* @pypy__format(%structtype_pypy.rlib.rbigint.rbigint* %a_1, %structtype_rpy_string* %digits_0, %structtype_rpy_string* %prefix_3, %structtype_rpy_string* %suffix_0) {
block0:
	br i1 false, label %block67, label %block13

block13:		; preds = %block0
	ret %structtype_rpy_string* null

block31:		; preds = %block67, %block44
	ret %structtype_rpy_string* null

block42:		; preds = %block67, %block44
	%j_167.reg2mem.0 = phi i32 [ %v63822, %block44 ], [ 0, %block67 ]		; <i32> [#uses=1]
	%v63822 = add i32 %j_167.reg2mem.0, -1		; <i32> [#uses=3]
	%v63823 = icmp slt i32 %v63822, 0		; <i1> [#uses=1]
	br i1 %v63823, label %block46, label %block43

block43:		; preds = %block42
	br label %block44

block44:		; preds = %block46, %block43
	%v6377959 = icmp sgt i32 %v63822, 0		; <i1> [#uses=1]
	br i1 %v6377959, label %block42, label %block31

block46:		; preds = %block42
	br label %block44

block67:		; preds = %block0
	br i1 false, label %block42, label %block31
}
