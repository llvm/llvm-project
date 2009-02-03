; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f

; bitcast a i64 to v2i32

define void @convert(<2 x i32>* %dst.addr, i64 %src) nounwind {
entry:
	%conv = bitcast i64 %src to <2 x i32>
	%xor = xor <2 x i32> %conv, < i32 255, i32 32767 >
	store <2 x i32> %xor, <2 x i32>* %dst.addr
	ret void
}
