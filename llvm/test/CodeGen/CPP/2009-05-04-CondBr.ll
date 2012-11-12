; RUN: llc < %s -march=cpp -cppgen=program -o %t
; RUN: grep "BranchInst::Create(label_if_then, label_if_end, int1_cmp, label_entry);" %t

define i32 @some_func(i32 %a) nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%a.addr = alloca i32		; <i32*> [#uses=8]
	store i32 %a, i32* %a.addr
	%tmp = load i32* %a.addr		; <i32> [#uses=1]
	%inc = add i32 %tmp, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %a.addr
	%tmp1 = load i32* %a.addr		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp1, 3		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	store i32 7, i32* %a.addr
	br label %if.end

if.end:		; preds = %if.then, %entry
	%tmp2 = load i32* %a.addr		; <i32> [#uses=1]
	%inc3 = add i32 %tmp2, 1		; <i32> [#uses=1]
	store i32 %inc3, i32* %a.addr
	%tmp4 = load i32* %a.addr		; <i32> [#uses=1]
	store i32 %tmp4, i32* %retval
	%0 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %0
}
