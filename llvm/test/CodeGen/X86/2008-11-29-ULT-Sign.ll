; RUN:  llvm-as < %s | llc -mtriple=i686-pc-linux-gnu | grep "jns" | count 1
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

define i32 @a(i32 %x) nounwind {
entry:
	%cmp = icmp ult i32 %x, -2147483648		; <i1> [#uses=1]
	br i1 %cmp, label %if.end, label %if.then

if.then:		; preds = %entry
	%call = call i32 (...)* @b()		; <i32> [#uses=0]
	br label %if.end

if.end:		; preds = %if.then, %entry
	br label %return

return:		; preds = %if.end
	ret i32 undef
}

declare i32 @b(...)

