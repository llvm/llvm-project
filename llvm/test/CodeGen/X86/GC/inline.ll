; RUN: opt < %s -passes='cgscc(inline)' -S | grep example

	%IntArray = type { i32, [0 x ptr] }

declare void @llvm.gcroot(ptr, ptr) nounwind

define i32 @f() {
	%x = call i32 @g( )		; <i32> [#uses=1]
	ret i32 %x
}

define internal i32 @g() gc "example" {
	%root = alloca ptr		; <ptr> [#uses=2]
	call void @llvm.gcroot( ptr %root, ptr null )
	%obj = call ptr @h( )		; <ptr> [#uses=2]
	store ptr %obj, ptr %root
	%Length.ptr = getelementptr %IntArray, ptr %obj, i32 0, i32 0		; <ptr> [#uses=1]
	%Length = load i32, ptr %Length.ptr		; <i32> [#uses=1]
	ret i32 %Length
}

declare ptr @h()
