; RUN: %lli -jit-kind=mcjit -disable-lazy-compilation=false %s

define i32 @main() nounwind {
entry:
	call void @lazily_compiled_address_is_consistent()
	ret i32 0
}

; Test PR3043: @test should have the same address before and after
; it's JIT-compiled.
@funcPtr = common global ptr null, align 4
@lcaic_failure = internal constant [46 x i8] c"@lazily_compiled_address_is_consistent failed\00"

define void @lazily_compiled_address_is_consistent() nounwind {
entry:
	store ptr @test, ptr @funcPtr
	%pass = tail call i1 @test()		; <i32> [#uses=1]
	br i1 %pass, label %pass_block, label %fail_block
pass_block:
	ret void
fail_block:
	call i32 @puts(ptr @lcaic_failure)
	call void @exit(i32 1)
	unreachable
}

define i1 @test() nounwind {
entry:
	%tmp = load ptr, ptr @funcPtr
	%eq = icmp eq ptr %tmp, @test
	ret i1 %eq
}

declare i32 @puts(ptr) noreturn
declare void @exit(i32) noreturn
