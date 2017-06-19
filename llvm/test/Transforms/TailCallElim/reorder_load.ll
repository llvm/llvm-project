; RUN: opt < %s -tailcallelim -S | FileCheck %s
; PR4323

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Several cases where tail call elimination should move the load above the call,
; then eliminate the tail recursion.



@global = external global i32		; <i32*> [#uses=1]
@extern_weak_global = extern_weak global i32		; <i32*> [#uses=1]


; This load can be moved above the call because the function won't write to it
; and the call has no side effects.
define fastcc i32 @raise_load_1(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind readonly {
; CHECK-LABEL: @raise_load_1(
; CHECK-NOT: call
; CHECK: load i32, i32*
; CHECK-NOT: call
; CHECK: }
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_1(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32, i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; This load can be moved above the call because the function won't write to it
; and the load provably can't trap.
define fastcc i32 @raise_load_2(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
; CHECK-LABEL: @raise_load_2(
; CHECK-NOT: call
; CHECK: load i32, i32*
; CHECK-NOT: call
; CHECK: }
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%nullcheck = icmp eq i32* %a_arg, null		; <i1> [#uses=1]
	br i1 %nullcheck, label %unwind, label %recurse

unwind:		; preds = %else
	unreachable

recurse:		; preds = %else
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_2(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32, i32* @global		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; This load can be safely moved above the call (even though it's from an
; extern_weak global) because the call has no side effects.
define fastcc i32 @raise_load_3(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind readonly {
; CHECK-LABEL: @raise_load_3(
; CHECK-NOT: call
; CHECK: load i32, i32*
; CHECK-NOT: call
; CHECK: }
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_3(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32, i32* @extern_weak_global		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; The second load can be safely moved above the call even though it's from an
; unknown pointer (which normally means it might trap) because the first load
; proves it doesn't trap.
define fastcc i32 @raise_load_4(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
; CHECK-LABEL: @raise_load_4(
; CHECK-NOT: call
; CHECK: load i32, i32*
; CHECK-NEXT: load i32, i32*
; CHECK-NOT: call
; CHECK: }
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%nullcheck = icmp eq i32* %a_arg, null		; <i1> [#uses=1]
	br i1 %nullcheck, label %unwind, label %recurse

unwind:		; preds = %else
	unreachable

recurse:		; preds = %else
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%first = load i32, i32* %a_arg		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_4(i32* %a_arg, i32 %first, i32 %tmp7)		; <i32> [#uses=1]
	%second = load i32, i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %second, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}

; This load can be moved above the call because the function won't write to it
; and the a_arg is dereferenceable.
define fastcc i32 @raise_load_5(i32* dereferenceable(4) %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
; CHECK-LABEL: @raise_load_5(
; CHECK-NOT: call
; CHECK: load i32, i32*
; CHECK-NOT: call
; CHECK: }
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_5(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32, i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}

; This load can be moved above the call because the function call does not write to the memory the load
; is accessing and the load is safe to speculate.
define fastcc i32 @raise_load_6(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind  {
; CHECK-LABEL: @raise_load_6(
; CHECK-NOT: call
; CHECK: load i32, i32*
; CHECK-NOT: call
; CHECK: }
entry:
  %s = alloca i32
  store i32 4, i32* %s
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  store i32 1, i32* %a_arg
	ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_6(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32, i32* %s		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}
