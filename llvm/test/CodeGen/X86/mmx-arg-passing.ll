; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -mattr=+mmx | grep mm0 | count 3
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -mattr=+mmx | grep esp | count 1
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep xmm0
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep rdi
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | not grep movups
;
; On Darwin x86-32, v8i8, v4i16, v2i32 values are passed in MM[0-2].
; On Darwin x86-32, v1i64 values are passed in memory.
; On Darwin x86-64, v8i8, v4i16, v2i32 values are passed in XMM[0-7].
; On Darwin x86-64, v1i64 values are passed in 64-bit GPRs.

@u1 = external global <8 x i8>

define void @t1(<8 x i8> %v1) nounwind  {
	store <8 x i8> %v1, <8 x i8>* @u1, align 8
	ret void
}

@u2 = external global <1 x i64>

define void @t2(<1 x i64> %v1) nounwind  {
	store <1 x i64> %v1, <1 x i64>* @u2, align 8
	ret void
}
