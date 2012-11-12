; RUN: opt < %s -scalarrepl -S | not grep alloca
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
	%struct..0anon = type { <1 x i64> }

define i32 @main(i32 %argc, i8** %argv) {
entry:
	%c = alloca %struct..0anon		; <%struct..0anon*> [#uses=2]
	%tmp2 = getelementptr %struct..0anon* %c, i32 0, i32 0		; <<1 x i64>*> [#uses=1]
	store <1 x i64> zeroinitializer, <1 x i64>* %tmp2, align 8
	%tmp7 = getelementptr %struct..0anon* %c, i32 0, i32 0		; <<1 x i64>*> [#uses=1]
	%tmp78 = bitcast <1 x i64>* %tmp7 to [2 x i32]*		; <[2 x i32]*> [#uses=1]
	%tmp9 = getelementptr [2 x i32]* %tmp78, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp10 = load i32* %tmp9, align 4		; <i32> [#uses=0]
	unreachable
}
