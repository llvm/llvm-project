; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {phi i32} | count 2

define void @test() nounwind  {
entry:
	br label %bb

bb:		; preds = %bb16, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %indvar.next, %somebb ]		; <i32> [#uses=1]
	%x.0 = phi i32 [ 37, %entry ], [ %tmp17, %somebb ]		; <i32> [#uses=1]
	%tmp = tail call i32 (...)* @bork( ) nounwind 		; <i32> [#uses=0]
	%tmp1 = tail call i32 (...)* @bork( ) nounwind 		; <i32> [#uses=0]
	%tmp2 = tail call i32 (...)* @bork( ) nounwind 		; <i32> [#uses=1]
	%tmp3 = icmp eq i32 %tmp2, 0		; <i1> [#uses=1]
	br i1 %tmp3, label %bb7, label %bb5

bb5:		; preds = %bb
	%tmp6 = tail call i32 (...)* @bork( ) nounwind 		; <i32> [#uses=0]
	br label %bb7

bb7:		; preds = %bb5, %bb
	%tmp8 = tail call i32 (...)* @bork( ) nounwind 		; <i32> [#uses=0]
	%tmp9 = tail call i32 (...)* @bork( ) nounwind 		; <i32> [#uses=0]
	%tmp11 = icmp eq i32 %x.0, 37		; <i1> [#uses=1]
	br i1 %tmp11, label %bb14, label %bb16

bb14:		; preds = %bb7
	%tmp15 = tail call i32 (...)* @bar( ) nounwind 		; <i32> [#uses=0]
	br label %bb16

bb16:		; preds = %bb14, %bb7
	%tmp17 = tail call i32 (...)* @zap( ) nounwind 		; <i32> [#uses=1]
	%indvar.next = add i32 %i.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 42		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %somebb

somebb:
	br label %bb

return:		; preds = %bb16
	ret void
}

declare i32 @bork(...)

declare i32 @bar(...)

declare i32 @zap(...)
