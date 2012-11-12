; RUN: llc < %s
; rdar://6836460
; rdar://7516906
; PR5963

define i32 @test(i128* %P) nounwind {
entry:
	%tmp48 = load i128* %P
	%and49 = and i128 %tmp48, 18446744073709551616		; <i128> [#uses=1]
	%tobool = icmp ne i128 %and49, 0		; <i1> [#uses=1]
	br i1 %tobool, label %if.then50, label %if.end61

if.then50:		; preds = %if.then20
	ret i32 1241

if.end61:		; preds = %if.then50, %if.then20, %entry
	ret i32 123
}

define i32 @test2(i320* %P) nounwind {
entry:
	%tmp48 = load i320* %P
	%and49 = and i320 %tmp48, 25108406941546723055343157692830665664409421777856138051584
	%tobool = icmp ne i320 %and49, 0		; <i1> [#uses=1]
	br i1 %tobool, label %if.then50, label %if.end61

if.then50:		; preds = %if.then20
	ret i32 1241

if.end61:		; preds = %if.then50, %if.then20, %entry
	ret i32 123
}
