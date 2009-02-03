; RUN: llvm-as < %s | \
; RUN:   opt -predsimplify -instcombine -simplifycfg | llvm-dis > %t
; RUN: grep -v declare %t | not grep fail
; RUN: grep -v declare %t | grep pass | count 4


define void @test1(i32 %x) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %A, label %then.1, label %else.1
then.1:		; preds = %entry
	%B = icmp eq i32 %x, 1		; <i1> [#uses=1]
	br i1 %B, label %then.2, label %else.1
then.2:		; preds = %then.1
	call void (...)* @fail( )
	ret void
else.1:		; preds = %then.1, %entry
	ret void
}

define void @test2(i32 %x) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	%B = icmp eq i32 %x, 1		; <i1> [#uses=1]
	br i1 %A, label %then.1, label %else.1
then.1:		; preds = %entry
	br i1 %B, label %then.2, label %else.1
then.2:		; preds = %then.1
	call void (...)* @fail( )
	ret void
else.1:		; preds = %then.1, %entry
	ret void
}

define void @test3(i32 %x) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	%B = icmp eq i32 %x, 1		; <i1> [#uses=1]
	br i1 %A, label %then.1, label %else.1
then.1:		; preds = %entry
	br i1 %B, label %then.2, label %else.1
then.2:		; preds = %then.1
	call void (...)* @fail( )
	ret void
else.1:		; preds = %then.1, %entry
	ret void
}

define void @test4(i32 %x, i32 %y) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	%B = icmp eq i32 %y, 0		; <i1> [#uses=1]
	%C = and i1 %A, %B		; <i1> [#uses=1]
	br i1 %C, label %then.1, label %else.1
then.1:		; preds = %entry
	%D = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %D, label %then.2, label %else.2
then.2:		; preds = %then.1
	%E = icmp eq i32 %y, 0		; <i1> [#uses=1]
	br i1 %E, label %else.1, label %else.2
else.1:		; preds = %then.2, %entry
	ret void
else.2:		; preds = %then.2, %then.1
	call void (...)* @fail( )
	ret void
}

define void @test5(i32 %x) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %A, label %then.1, label %else.1
then.1:		; preds = %else.1, %entry
	ret void
then.2:		; preds = %else.1
	call void (...)* @fail( )
	ret void
else.1:		; preds = %entry
	%B = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %B, label %then.2, label %then.1
}

define void @test6(i32 %x, i32 %y) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	%B = icmp eq i32 %y, 0		; <i1> [#uses=1]
	%C = or i1 %A, %B		; <i1> [#uses=1]
	br i1 %C, label %then.1, label %else.1
then.1:		; preds = %else.2, %entry
	ret void
then.2:		; preds = %else.2, %else.1
	call void (...)* @fail( )
	ret void
else.1:		; preds = %entry
	%D = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %D, label %then.2, label %else.2
else.2:		; preds = %else.1
	%E = icmp ne i32 %y, 0		; <i1> [#uses=1]
	br i1 %E, label %then.1, label %then.2
}

define void @test7(i32 %x) {
entry:
	%A = icmp ne i32 %x, 0		; <i1> [#uses=1]
	%B = xor i1 %A, true		; <i1> [#uses=1]
	br i1 %B, label %then.1, label %else.1
then.1:		; preds = %entry
	%C = icmp eq i32 %x, 1		; <i1> [#uses=1]
	br i1 %C, label %then.2, label %else.1
then.2:		; preds = %then.1
	call void (...)* @fail( )
	ret void
else.1:		; preds = %then.1, %entry
	ret void
}

define void @test8(i32 %x) {
entry:
	%A = add i32 %x, 1		; <i32> [#uses=1]
	%B = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %B, label %then.1, label %then.2
then.1:		; preds = %entry
	%C = icmp eq i32 %A, 1		; <i1> [#uses=1]
	br i1 %C, label %then.2, label %else.2
then.2:		; preds = %then.1, %entry
	ret void
else.2:		; preds = %then.1
	call void (...)* @fail( )
	ret void
}

define void @test9(i32 %y, i32 %z) {
entry:
	%x = add i32 %y, %z		; <i32> [#uses=1]
	%A = icmp eq i32 %y, 3		; <i1> [#uses=1]
	%B = icmp eq i32 %z, 5		; <i1> [#uses=1]
	%C = and i1 %A, %B		; <i1> [#uses=1]
	br i1 %C, label %cond_true, label %return
cond_true:		; preds = %entry
	%D = icmp eq i32 %x, 8		; <i1> [#uses=1]
	br i1 %D, label %then, label %oops
then:		; preds = %cond_true
	call void (...)* @pass( )
	ret void
oops:		; preds = %cond_true
	call void (...)* @fail( )
	ret void
return:		; preds = %entry
	ret void
}

define void @test10() {
entry:
	%A = alloca i32		; <i32*> [#uses=1]
	%B = icmp eq i32* %A, null		; <i1> [#uses=1]
	br i1 %B, label %cond_true, label %cond_false
cond_true:		; preds = %entry
	call void (...)* @fail( )
	ret void
cond_false:		; preds = %entry
	call void (...)* @pass( )
	ret void
}

define void @switch1(i32 %x) {
entry:
	%A = icmp eq i32 %x, 10		; <i1> [#uses=1]
	br i1 %A, label %return, label %cond_false
cond_false:		; preds = %entry
	switch i32 %x, label %return [
		 i32 9, label %then1
		 i32 10, label %then2
	]
then1:		; preds = %cond_false
	call void (...)* @pass( )
	ret void
then2:		; preds = %cond_false
	call void (...)* @fail( )
	ret void
return:		; preds = %cond_false, %entry
	ret void
}

define void @switch2(i32 %x) {
entry:
	%A = icmp eq i32 %x, 10		; <i1> [#uses=1]
	br i1 %A, label %return, label %cond_false
cond_false:		; preds = %entry
	switch i32 %x, label %return [
		 i32 8, label %then1
		 i32 9, label %then1
		 i32 10, label %then1
	]
then1:		; preds = %cond_false, %cond_false, %cond_false
	%B = icmp ne i32 %x, 8		; <i1> [#uses=1]
	br i1 %B, label %then2, label %return
then2:		; preds = %then1
	call void (...)* @pass( )
	ret void
return:		; preds = %then1, %cond_false, %entry
	ret void
}

define void @switch3(i32 %x) {
entry:
	%A = icmp eq i32 %x, 10		; <i1> [#uses=1]
	br i1 %A, label %return, label %cond_false
cond_false:		; preds = %entry
	switch i32 %x, label %return [
		 i32 9, label %then1
		 i32 10, label %then1
	]
then1:		; preds = %cond_false, %cond_false
	%B = icmp eq i32 %x, 9		; <i1> [#uses=1]
	br i1 %B, label %return, label %oops
oops:		; preds = %then1
	call void (...)* @fail( )
	ret void
return:		; preds = %then1, %cond_false, %entry
	ret void
}

define void @switch4(i32 %x) {
entry:
	%A = icmp eq i32 %x, 10		; <i1> [#uses=1]
	br i1 %A, label %then1, label %cond_false
cond_false:		; preds = %entry
	switch i32 %x, label %default [
		 i32 9, label %then1
		 i32 10, label %then2
	]
then1:		; preds = %default, %cond_false, %entry
	ret void
then2:		; preds = %cond_false
	ret void
default:		; preds = %cond_false
	%B = icmp eq i32 %x, 9		; <i1> [#uses=1]
	br i1 %B, label %oops, label %then1
oops:		; preds = %default
	call void (...)* @fail( )
	ret void
}

define void @select1(i32 %x) {
entry:
	%A = icmp eq i32 %x, 10		; <i1> [#uses=3]
	%B = select i1 %A, i32 1, i32 2		; <i32> [#uses=1]
	%C = icmp eq i32 %B, 1		; <i1> [#uses=1]
	br i1 %C, label %then, label %else
then:		; preds = %entry
	br i1 %A, label %return, label %oops
else:		; preds = %entry
	br i1 %A, label %oops, label %return
oops:		; preds = %else, %then
	call void (...)* @fail( )
	ret void
return:		; preds = %else, %then
	ret void
}

define void @select2(i32 %x) {
entry:
	%A = icmp eq i32 %x, 10		; <i1> [#uses=2]
	%B = select i1 %A, i32 1, i32 2		; <i32> [#uses=1]
	%C = icmp eq i32 %B, 1		; <i1> [#uses=2]
	br i1 %A, label %then, label %else
then:		; preds = %entry
	br i1 %C, label %return, label %oops
else:		; preds = %entry
	br i1 %C, label %oops, label %return
oops:		; preds = %else, %then
	call void (...)* @fail( )
	ret void
return:		; preds = %else, %then
	ret void
}

declare void @fail(...)

declare void @pass(...)
