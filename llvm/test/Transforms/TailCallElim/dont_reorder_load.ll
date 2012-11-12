; RUN: opt < %s -tailcallelim -S | grep call | count 3
; PR4323

; Several cases where tail call elimination should not move the load above the
; call, and thus can't eliminate the tail recursion.


@extern_weak_global = extern_weak global i32		; <i32*> [#uses=1]


; This load can't be safely moved above the call because the load is from an
; extern_weak global and may trap, but the call may unwind before that happens.
define fastcc i32 @no_tailrecelim_1(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	ret i32 37

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @no_tailrecelim_1(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32* @extern_weak_global		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}


; This load can't be safely moved above the call because function may write to the pointer.
define fastcc i32 @no_tailrecelim_2(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind {
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
	store i32 1, i32* %a_arg
        ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @no_tailrecelim_2(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}

; This load can't be safely moved above the call because that would change the
; order in which the load volatiles are performed.
define fastcc i32 @no_tailrecelim_3(i32* %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind {
entry:
	%tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
	br i1 %tmp2, label %if, label %else

if:		; preds = %entry
        ret i32 0

else:		; preds = %entry
	%tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
	%tmp8 = call fastcc i32 @no_tailrecelim_3(i32* %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
	%tmp9 = load volatile i32* %a_arg		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
	ret i32 %tmp10
}
