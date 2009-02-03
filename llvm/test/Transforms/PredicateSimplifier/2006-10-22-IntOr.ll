; RUN: llvm-as < %s | \
; RUN:   opt -predsimplify -instcombine -simplifycfg  | llvm-dis > %t
; RUN: grep -v declare %t | not grep fail
; RUN: grep -v declare %t | grep pass | count 3

define i32 @test1(i32 %x, i32 %y) {
entry:
	%tmp2 = or i32 %x, %y		; <i32> [#uses=1]
	%tmp = icmp eq i32 %tmp2, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %return
cond_true:		; preds = %entry
	%tmp4 = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %cond_true5, label %cond_false
cond_true5:		; preds = %cond_true
	%tmp6 = call i32 @pass( )		; <i32> [#uses=1]
	ret i32 %tmp6
cond_false:		; preds = %cond_true
	%tmp8 = call i32 @fail( )		; <i32> [#uses=1]
	ret i32 %tmp8
return:		; preds = %entry
	ret i32 0
}

define i32 @test2(i32 %x, i32 %y) {
entry:
	%tmp2 = or i32 %x, %y		; <i32> [#uses=1]
	%tmp = icmp ne i32 %tmp2, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %return
cond_true:		; preds = %entry
	%tmp4 = icmp eq i32 %x, 0		; <i1> [#uses=1]
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

declare i32 @fail()

declare i32 @pass()

declare i32 @pass1()

declare i32 @pass2()
