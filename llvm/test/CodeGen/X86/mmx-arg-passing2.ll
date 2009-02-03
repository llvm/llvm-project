; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep movq2dq | count 1
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep movdq2q | count 2

@g_v8qi = external global <8 x i8>

define void @t1() nounwind  {
	%tmp3 = load <8 x i8>* @g_v8qi, align 8
	%tmp4 = tail call i32 (...)* @pass_v8qi( <8 x i8> %tmp3 ) nounwind
	ret void
}

define void @t2(<8 x i8> %v1, <8 x i8> %v2) nounwind  {
       %tmp3 = add <8 x i8> %v1, %v2
       %tmp4 = tail call i32 (...)* @pass_v8qi( <8 x i8> %tmp3 ) nounwind
       ret void
}

define void @t3() nounwind  {
	call void @pass_v1di( <1 x i64> zeroinitializer )
        ret void
}

declare i32 @pass_v8qi(...)
declare void @pass_v1di(<1 x i64>)
