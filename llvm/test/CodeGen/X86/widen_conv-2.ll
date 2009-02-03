; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f

; sign extension v2i32 to v2i16

define void @convert(<2 x i32>* %dst.addr, <2 x i16> %src) nounwind {
entry:
	%signext = sext <2 x i16> %src to <2 x i32>		; <<12 x i8>> [#uses=1]
	store <2 x i32> %signext, <2 x i32>* %dst.addr
	ret void
}
