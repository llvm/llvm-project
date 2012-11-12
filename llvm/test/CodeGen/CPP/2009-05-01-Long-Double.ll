; RUN: llc < %s -march=cpp -cppgen=program -o %t

define x86_fp80 @some_func() nounwind {
entry:
	%retval = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%call = call i32 (...)* @other_func()		; <i32> [#uses=1]
	%conv = sitofp i32 %call to x86_fp80		; <x86_fp80> [#uses=1]
	store x86_fp80 %conv, x86_fp80* %retval
	%0 = load x86_fp80* %retval		; <x86_fp80> [#uses=1]
	ret x86_fp80 %0
}

declare i32 @other_func(...)
