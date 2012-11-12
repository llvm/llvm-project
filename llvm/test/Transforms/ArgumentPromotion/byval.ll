; RUN: opt < %s -argpromotion -scalarrepl -S | not grep load
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"
; Argpromote + scalarrepl should change this to passing the two integers by value.

	%struct.ss = type { i32, i64 }

define internal void @f(%struct.ss* byval  %b) nounwind  {
entry:
	%tmp = getelementptr %struct.ss* %b, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp1 = load i32* %tmp, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	store i32 %tmp2, i32* %tmp, align 4
	ret void
}

define i32 @main() nounwind  {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
	store i64 2, i64* %tmp4, align 4
	call void @f( %struct.ss* byval  %S ) nounwind 
	ret i32 0
}
