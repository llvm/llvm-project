; RUN: opt < %s -passes=instcombine -S | grep "ret i32 %A"
; RUN: opt < %s -dce -S | not grep call.*llvm

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

declare void @llvm.lifetime.start.p0i8(i64, i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8*)

define void @test3() {
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* undef)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* undef)
  ret void
}

