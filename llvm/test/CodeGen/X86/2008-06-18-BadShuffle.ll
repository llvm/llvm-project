; RUN: llvm-as < %s | llc -march=x86 -mcpu=i386 -mattr=+sse2 | grep pinsrw

; Test to make sure we actually insert the bottom element of the vector
define <8 x i16> @a(<8 x i16> %a) nounwind  {
entry:
	shufflevector <8 x i16> %a, <8 x i16> zeroinitializer, <8 x i32> < i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8 >
	%add = add <8 x i16> %0, %a
	ret <8 x i16> %add
}

