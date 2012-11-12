; RUN: llc < %s
; PR2988
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin10.0"
@a = common global ppc_fp128 0xM00000000000000000000000000000000, align 16		; <ppc_fp128*> [#uses=2]
@b = common global ppc_fp128 0xM00000000000000000000000000000000, align 16		; <ppc_fp128*> [#uses=2]
@c = common global ppc_fp128 0xM00000000000000000000000000000000, align 16		; <ppc_fp128*> [#uses=3]
@d = common global ppc_fp128 0xM00000000000000000000000000000000, align 16		; <ppc_fp128*> [#uses=2]

define void @foo() nounwind {
entry:
	%0 = load ppc_fp128* @a, align 16		; <ppc_fp128> [#uses=1]
	%1 = call ppc_fp128 @llvm.sqrt.ppcf128(ppc_fp128 %0)		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %1, ppc_fp128* @a, align 16
	%2 = load ppc_fp128* @b, align 16		; <ppc_fp128> [#uses=1]
	%3 = call ppc_fp128 @"\01_sinl$LDBL128"(ppc_fp128 %2) nounwind readonly		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %3, ppc_fp128* @b, align 16
	%4 = load ppc_fp128* @c, align 16		; <ppc_fp128> [#uses=1]
	%5 = call ppc_fp128 @"\01_cosl$LDBL128"(ppc_fp128 %4) nounwind readonly		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %5, ppc_fp128* @c, align 16
	%6 = load ppc_fp128* @d, align 16		; <ppc_fp128> [#uses=1]
	%7 = load ppc_fp128* @c, align 16		; <ppc_fp128> [#uses=1]
	%8 = call ppc_fp128 @llvm.pow.ppcf128(ppc_fp128 %6, ppc_fp128 %7)		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %8, ppc_fp128* @d, align 16
	br label %return

return:		; preds = %entry
	ret void
}

declare ppc_fp128 @llvm.sqrt.ppcf128(ppc_fp128) nounwind readonly

declare ppc_fp128 @"\01_sinl$LDBL128"(ppc_fp128) nounwind readonly

declare ppc_fp128 @"\01_cosl$LDBL128"(ppc_fp128) nounwind readonly

declare ppc_fp128 @llvm.pow.ppcf128(ppc_fp128, ppc_fp128) nounwind readonly

declare ppc_fp128 @copysignl(ppc_fp128, ppc_fp128)

define ppc_fp128 @cs(ppc_fp128 %from, ppc_fp128 %to) {
  %tmp = call ppc_fp128 @copysignl(ppc_fp128 %from, ppc_fp128 %to)
  ret ppc_fp128 %tmp
}
