; RUN: llvm-as < %s | \
; RUN:   opt -predsimplify -instcombine -simplifycfg | llvm-dis | \
; RUN:   grep -v declare | grep pass | count 2

define i32 @test(i32 %x, i32 %y) {
entry:
	%tmp2 = icmp slt i32 %x, %y		; <i1> [#uses=1]
	%tmp = icmp ne i1 %tmp2, true		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %return
cond_true:		; preds = %entry
	%tmp4 = icmp eq i32 %x, %y		; <i1> [#uses=1]
	br i1 %tmp4, label %cond_true5, label %cond_false
cond_true5:		; preds = %cond_true
	%tmp6 = call i32 @pass1( )		; <i32> [#uses=1]
	ret i32 %tmp6
cond_false:		; preds = %cond_true
	%tmp8 = call i32 @pass2( )		; <i32> [#uses=1]
	ret i32 %tmp8
return:		; preds = %entry
	ret i32 0
}

declare i32 @pass1()

declare i32 @pass2()

