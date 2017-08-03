; RUN: llc < %s -march=x86 -mtriple=x86_64-unknown-freebsd7.2
; PR4478

	%struct.sockaddr = type <{ i8, i8, [14 x i8] }>

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br label %while.cond

while.cond:		; preds = %sw.bb6, %entry
	switch i32 undef, label %sw.default [
		i32 -1, label %while.end
		i32 119, label %sw.bb6
	]

sw.bb6:		; preds = %while.cond
	br i1 undef, label %if.then, label %while.cond

if.then:		; preds = %sw.bb6
	ret i32 1

sw.default:		; preds = %while.cond
	ret i32 1

while.end:		; preds = %while.cond
	br i1 undef, label %if.then15, label %if.end16

if.then15:		; preds = %while.end
	ret i32 1

if.end16:		; preds = %while.end
	br i1 undef, label %lor.lhs.false, label %if.then21

lor.lhs.false:		; preds = %if.end16
	br i1 undef, label %if.end22, label %if.then21

if.then21:		; preds = %lor.lhs.false, %if.end16
	ret i32 1

if.end22:		; preds = %lor.lhs.false
	br i1 undef, label %lor.lhs.false27, label %if.then51

lor.lhs.false27:		; preds = %if.end22
	br i1 undef, label %lor.lhs.false39, label %if.then51

lor.lhs.false39:		; preds = %lor.lhs.false27
	br i1 undef, label %if.end52, label %if.then51

if.then51:		; preds = %lor.lhs.false39, %lor.lhs.false27, %if.end22
	ret i32 1

if.end52:		; preds = %lor.lhs.false39
	br i1 undef, label %if.then57, label %if.end58

if.then57:		; preds = %if.end52
	ret i32 1

if.end58:		; preds = %if.end52
	br i1 undef, label %if.then64, label %if.end65

if.then64:		; preds = %if.end58
	ret i32 1

if.end65:		; preds = %if.end58
	br i1 undef, label %if.then71, label %if.end72

if.then71:		; preds = %if.end65
	ret i32 1

if.end72:		; preds = %if.end65
	br i1 undef, label %if.then83, label %if.end84

if.then83:		; preds = %if.end72
	ret i32 1

if.end84:		; preds = %if.end72
	br i1 undef, label %if.then101, label %if.end102

if.then101:		; preds = %if.end84
	ret i32 1

if.end102:		; preds = %if.end84
	br i1 undef, label %if.then113, label %if.end114

if.then113:		; preds = %if.end102
	ret i32 1

if.end114:		; preds = %if.end102
	br i1 undef, label %if.then209, label %if.end210

if.then209:		; preds = %if.end114
	ret i32 1

if.end210:		; preds = %if.end114
	br i1 undef, label %if.then219, label %if.end220

if.then219:		; preds = %if.end210
	ret i32 1

if.end220:		; preds = %if.end210
	br i1 undef, label %if.end243, label %lor.lhs.false230

lor.lhs.false230:		; preds = %if.end220
	unreachable

if.end243:		; preds = %if.end220
	br i1 undef, label %if.then249, label %if.end250

if.then249:		; preds = %if.end243
	ret i32 1

if.end250:		; preds = %if.end243
	br i1 undef, label %if.end261, label %if.then260

if.then260:		; preds = %if.end250
	ret i32 1

if.end261:		; preds = %if.end250
	br i1 undef, label %if.then270, label %if.end271

if.then270:		; preds = %if.end261
	ret i32 1

if.end271:		; preds = %if.end261
	%call.i = call i32 @arc4random() nounwind		; <i32> [#uses=1]
	%rem.i = urem i32 %call.i, 16383		; <i32> [#uses=1]
	%rem1.i = trunc i32 %rem.i to i16		; <i16> [#uses=1]
	%conv2.i = or i16 %rem1.i, -16384		; <i16> [#uses=1]
	%0 = call i16 asm "xchgb ${0:h}, ${0:b}", "=Q,0,~{dirflag},~{fpsr},~{flags}"(i16 %conv2.i) nounwind		; <i16> [#uses=1]
	store i16 %0, i16* undef
	%call281 = call i32 @bind(i32 undef, %struct.sockaddr* undef, i32 16) nounwind		; <i32> [#uses=0]
	unreachable
}

declare i32 @bind(i32, %struct.sockaddr*, i32)

declare i32 @arc4random()
