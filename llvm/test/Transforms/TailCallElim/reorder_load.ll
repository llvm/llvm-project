; RUN: opt < %s -tailcallelim -S | not grep call
; PR4323

; Several cases where tail call elimination should move the load above the call,
; then eliminate the tail recursion.


@global = external global i32		; <i32*> [#uses=1]
@extern_weak_global = extern_weak global i32		; <i32*> [#uses=1]


; This load can be moved above the call because the function won't write to it
; and the call has no side effects.
define fastcc i32 @raise_load_1(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind readonly {
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_1(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; This load can be moved above the call because the function won't write to it
; and the load provably can't trap.
define fastcc i32 @raise_load_2(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
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
	%tmp9 = load i32* @global		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; This load can be safely moved above the call (even though it's from an
; extern_weak global) because the call has no side effects.
define fastcc i32 @raise_load_3(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind readonly {
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_3(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32* @extern_weak_global		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; The second load can be safely moved above the call even though it's from an
; unknown pointer (which normally means it might trap) because the first load
; proves it doesn't trap.
define fastcc i32 @raise_load_4(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
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
	%first = load i32* %a_arg		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @raise_load_4(i32* %a_arg, i32 %first, i32 %tmp7)		; <i32> [#uses=1]
	%second = load i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %second, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}
