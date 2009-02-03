; RUN: llvm-as < %s | opt -inline -argpromotion -disable-output

define internal fastcc i32 @hash(i32* %ts, i32 %mod) nounwind {
entry:
	unreachable
}

define void @encode(i32* %m, i32* %ts, i32* %new) nounwind {
entry:
	%0 = call fastcc i32 @hash( i32* %ts, i32 0 ) nounwind		; <i32> [#uses=0]
	unreachable
}
