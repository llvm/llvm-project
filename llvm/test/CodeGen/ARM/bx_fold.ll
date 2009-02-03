; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=arm | not grep bx

define void @test(i32 %Ptr, i8* %L) {
entry:
	br label %bb1

bb:		; preds = %bb1
	%gep.upgrd.1 = zext i32 %indvar to i64		; <i64> [#uses=1]
	%tmp7 = getelementptr i8* %L, i64 %gep.upgrd.1		; <i8*> [#uses=1]
	store i8 0, i8* %tmp7
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	%i.0 = bitcast i32 %indvar to i32		; <i32> [#uses=2]
	%tmp = tail call i32 (...)* @bar( )		; <i32> [#uses=1]
	%tmp2 = add i32 %i.0, %tmp		; <i32> [#uses=1]
	%Ptr_addr.0 = sub i32 %Ptr, %tmp2		; <i32> [#uses=0]
	%tmp12 = icmp eq i32 %i.0, %Ptr		; <i1> [#uses=1]
	%tmp12.not = xor i1 %tmp12, true		; <i1> [#uses=1]
	%bothcond = and i1 %tmp12.not, false		; <i1> [#uses=1]
	br i1 %bothcond, label %bb, label %bb18

bb18:		; preds = %bb1
	ret void
}

declare i32 @bar(...)
