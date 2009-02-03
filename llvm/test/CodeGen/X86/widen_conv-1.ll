; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f
; RUN: grep pshufd %t | count 1
; RUN: grep paddd  %t | count 1

; truncate v2i64 to v2i32

define void @convert(<2 x i32>* %dst.addr, <2 x i64> %src) nounwind {
entry:
	%val = trunc <2 x i64> %src to <2 x i32>
	%add = add <2 x i32> %val, < i32 1, i32 1 >
	store <2 x i32> %add, <2 x i32>* %dst.addr
	ret void
}
