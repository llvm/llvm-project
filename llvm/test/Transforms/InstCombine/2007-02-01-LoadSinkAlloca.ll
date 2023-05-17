; RUN: opt < %s -passes='function(instcombine),function(mem2reg)' -S | grep "%A = alloca" 
; RUN: opt < %s -passes='function(instcombine),function(mem2reg)' -S | \
; RUN:    not grep "%B = alloca"
; END.

; Ensure that instcombine doesn't sink the loads in entry/cond_true into 
; cond_next.  Doing so prevents mem2reg from promoting the B alloca.

define i32 @test2(i32 %C) {
entry:
	%A = alloca i32
	%B = alloca i32
	%tmp = call i32 (...) @bar( ptr %A )		; <i32> [#uses=0]
	%T = load i32, ptr %A		; <i32> [#uses=1]
	%tmp2 = icmp eq i32 %C, 0		; <i1> [#uses=1]
	br i1 %tmp2, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	store i32 123, ptr %B
	call i32 @test2( i32 123 )		; <i32>:0 [#uses=0]
	%T1 = load i32, ptr %B		; <i32> [#uses=1]
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp1.0 = phi i32 [ %T1, %cond_true ], [ %T, %entry ]		; <i32> [#uses=1]
	%tmp7 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp8 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp9 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp10 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp11 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp12 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp13 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp14 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp15 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp16 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp17 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp18 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp19 = call i32 (...) @baq( )		; <i32> [#uses=0]
	%tmp20 = call i32 (...) @baq( )		; <i32> [#uses=0]
	ret i32 %tmp1.0
}

declare i32 @bar(...)

declare i32 @baq(...)
