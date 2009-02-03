; RUN: llvm-as < %s | llc
; PR2623

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-unknown-freebsd7.0"
	%.objc_id = type { %.objc_id }*
	%.objc_selector = type { i8*, i8* }*
@.objc_sel_ptr = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr13 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr14 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr15 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr16 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr17 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr18 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr19 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr20 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]
@.objc_sel_ptr21 = external constant %.objc_selector		; <%.objc_selector*> [#uses=1]

@.objc_untyped_selector_alias = alias internal %.objc_selector* @.objc_sel_ptr15		; <%.objc_selector*> [#uses=0]
@.objc_untyped_selector_alias1 = alias internal %.objc_selector* @.objc_sel_ptr		; <%.objc_selector*> [#uses=0]
@.objc_untyped_selector_alias2 = alias internal %.objc_selector* @.objc_sel_ptr17		; <%.objc_selector*> [#uses=0]
@.objc_untyped_selector_alias3 = alias internal %.objc_selector* @.objc_sel_ptr16		; <%.objc_selector*> [#uses=0]
@.objc_untyped_selector_alias4 = alias internal %.objc_selector* @.objc_sel_ptr13		; <%.objc_selector*> [#uses=0]
@.objc_untyped_selector_alias7 = alias internal %.objc_selector* @.objc_sel_ptr14		; <%.objc_selector*> [#uses=0]
@getRange = alias internal %.objc_selector* @.objc_sel_ptr18		; <%.objc_selector*> [#uses=0]
@"valueWithRange:" = alias internal %.objc_selector* @.objc_sel_ptr21		; <%.objc_selector*> [#uses=0]
@rangeValue = alias internal %.objc_selector* @.objc_sel_ptr20		; <%.objc_selector*> [#uses=0]
@"printRange:" = alias internal %.objc_selector* @.objc_sel_ptr19		; <%.objc_selector*> [#uses=0]

define void @"._objc_method_SmalltalkTool()-run"(i8* %self, %.objc_selector %_cmd) {
entry:
	br i1 false, label %small_int_messagerangeValue, label %real_object_messagerangeValue

small_int_messagerangeValue:		; preds = %entry
	br label %Continue

real_object_messagerangeValue:		; preds = %entry
	br label %Continue

Continue:		; preds = %real_object_messagerangeValue, %small_int_messagerangeValue
	%rangeValue = phi { i32, i32 } [ undef, %small_int_messagerangeValue ], [ undef, %real_object_messagerangeValue ]		; <{ i32, i32 }> [#uses=1]
	call void (%.objc_id, %.objc_selector, ...)* null( %.objc_id null, %.objc_selector null, { i32, i32 } %rangeValue )
	ret void
}
