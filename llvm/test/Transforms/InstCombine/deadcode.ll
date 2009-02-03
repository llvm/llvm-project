; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i32 %A}
; RUN: llvm-as < %s | opt -die | llvm-dis | not grep call.*llvm.stacksave

define i32 @test(i32 %A) {
	%X = or i1 false, false		
	br i1 %X, label %T, label %C

T:		; preds = %0
	%B = add i32 %A, 1	
	br label %C

C:		; preds = %T, %0
	%C.upgrd.1 = phi i32 [ %B, %T ], [ %A, %0 ]
	ret i32 %C.upgrd.1
}

define i32* @test2(i32 %width) {
	%tmp = call i8* @llvm.stacksave( )
        %tmp14 = alloca i32, i32 %width
	ret i32* %tmp14
} 

declare i8* @llvm.stacksave()

