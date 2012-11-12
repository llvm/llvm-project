; RUN: opt < %s -argpromotion -S | grep zeroext

	%struct.ss = type { i32, i64 }

define internal void @f(%struct.ss* byval  %b, i32* byval %X, i32 %i) nounwind  {
entry:
	%tmp = getelementptr %struct.ss* %b, i32 0, i32 0
	%tmp1 = load i32* %tmp, align 4
	%tmp2 = add i32 %tmp1, 1	
	store i32 %tmp2, i32* %tmp, align 4

	store i32 0, i32* %X
	ret void
}

define i32 @test(i32* %X) {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
	store i64 2, i64* %tmp4, align 4
	call void @f( %struct.ss* byval %S, i32* byval %X, i32 zeroext 0) 
	ret i32 0
}
