; RUN: opt < %s -inline -S | grep example

	%IntArray = type { i32, [0 x i32*] }

declare void @llvm.gcroot(i8**, i8*) nounwind 

define i32 @f() {
	%x = call i32 @g( )		; <i32> [#uses=1]
	ret i32 %x
}

define internal i32 @g() gc "example" {
	%root = alloca i8*		; <i8**> [#uses=2]
	call void @llvm.gcroot( i8** %root, i8* null )
	%obj = call %IntArray* @h( )		; <%IntArray*> [#uses=2]
	%obj.2 = bitcast %IntArray* %obj to i8*		; <i8*> [#uses=1]
	store i8* %obj.2, i8** %root
	%Length.ptr = getelementptr %IntArray* %obj, i32 0, i32 0		; <i32*> [#uses=1]
	%Length = load i32* %Length.ptr		; <i32> [#uses=1]
	ret i32 %Length
}

declare %IntArray* @h()
