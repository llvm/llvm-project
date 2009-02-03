; RUN: llvm-as < %s | opt -predsimplify -disable-output
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define void @f(i32 %x, i32 %y) {
entry:
	%tmp = icmp eq i32 %x, 10		; <i1> [#uses=1]
	%tmp.not = xor i1 %tmp, true		; <i1> [#uses=1]
	%tmp3 = icmp eq i32 %x, %y		; <i1> [#uses=1]
	%bothcond = and i1 %tmp.not, %tmp3		; <i1> [#uses=1]
	br i1 %bothcond, label %cond_true4, label %return
cond_true4:		; preds = %entry
	switch i32 %y, label %return [
		 i32 9, label %bb
		 i32 10, label %bb6
	]
bb:		; preds = %cond_true4
	call void @g( i32 9 )
	ret void
bb6:		; preds = %cond_true4
	call void @g( i32 10 )
	ret void
return:		; preds = %cond_true4, %entry
	ret void
}

declare void @g(i32)

