; RUN: %lli -remote-mcjit -disable-lazy-compilation=false -relocation-model=pic -code-model=small %s
; XFAIL: *
; This function should fail until remote symbol resolution is supported.

define i32 @main() nounwind {
entry:
	call void @lazily_compiled_address_is_consistent()
	ret i32 0
}

; Test PR3043: @test should have the same address before and after
; it's JIT-compiled.
@funcPtr = common global i1 ()* null, align 4
@lcaic_failure = internal constant [46 x i8] c"@lazily_compiled_address_is_consistent failed\00"

define void @lazily_compiled_address_is_consistent() nounwind {
entry:
	store i1 ()* @test, i1 ()** @funcPtr
	%pass = tail call i1 @test()		; <i32> [#uses=1]
	br i1 %pass, label %pass_block, label %fail_block
pass_block:
	ret void
fail_block:
	call i32 @puts(i8* getelementptr([46 x i8], [46 x i8]* @lcaic_failure, i32 0, i32 0))
	call void @exit(i32 1)
	unreachable
}

define i1 @test() nounwind {
entry:
	%tmp = load i1 ()*, i1 ()** @funcPtr
	%eq = icmp eq i1 ()* %tmp, @test
	ret i1 %eq
}

declare i32 @puts(i8*) noreturn
declare void @exit(i32) noreturn
