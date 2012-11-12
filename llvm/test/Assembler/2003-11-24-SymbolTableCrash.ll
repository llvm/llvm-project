; RUN: not llvm-as < %s 2>&1 | grep "multiple definition"

define void @test() {
	%tmp.1 = add i32 0, 1
	br label %return
return:
	%tmp.1 = add i32 0, 1
	ret void
}

