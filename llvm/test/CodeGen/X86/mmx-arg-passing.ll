; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+mmx | grep mm0 | count 1
; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+mmx | grep esp | count 2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep xmm0
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | grep rdi
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | not grep movups
;
; On Darwin x86-32, v8i8, v4i16, v2i32 values are passed in MM[0-2].
; On Darwin x86-32, v1i64 values are passed in memory.  In this example, they
;                   are never moved into an MM register at all.
; On Darwin x86-64, v8i8, v4i16, v2i32 values are passed in XMM[0-7].
; On Darwin x86-64, v1i64 values are passed in 64-bit GPRs.

@u1 = external global x86_mmx

define void @t1(x86_mmx %v1) nounwind  {
	store x86_mmx %v1, x86_mmx* @u1, align 8
	ret void
}

@u2 = external global x86_mmx

define void @t2(<1 x i64> %v1) nounwind  {
        %tmp = bitcast <1 x i64> %v1 to x86_mmx
	store x86_mmx %tmp, x86_mmx* @u2, align 8
	ret void
}

