; RUN: llvm-as < %s | opt -licm -disable-output

define void @test({ i32 }* %P) {
	br label %Loop
Loop:		; preds = %Loop, %0
	free { i32 }* %P
	br label %Loop
}

