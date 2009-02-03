; RUN: llvm-as < %s | opt -dse | llvm-dis | grep {store i32 0}

define void @test({i32,i32 }* %P) {
	%Q = getelementptr {i32,i32}* %P, i32 1
	%X = getelementptr {i32,i32}* %Q, i32 0, i32 1
	%Y = getelementptr {i32,i32}* %Q, i32 1, i32 1
	store i32 0, i32* %X
	store i32 1, i32* %Y
	ret void
}
