; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: error: multiple definition of local value named 'tmp.1'
define void @test() {
	%tmp.1 = add i32 0, 1
	br label %return
return:
	%tmp.1 = add i32 0, 1
	ret void
}

