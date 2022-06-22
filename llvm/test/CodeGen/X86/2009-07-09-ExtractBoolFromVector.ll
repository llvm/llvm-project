; RUN: llc < %s -mtriple=i686--
; PR3037

define void @entry(ptr %dest) {
	%1 = xor <4 x i1> zeroinitializer, < i1 true, i1 true, i1 true, i1 true >
	%2 = extractelement <4 x i1> %1, i32 3
	%3 = zext i1 %2 to i8
	%4 = insertelement <4 x i8> zeroinitializer, i8 %3, i32 3
	store <4 x i8> %4, ptr %dest, align 1
	ret void
}
