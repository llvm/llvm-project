; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; pr4843

define <4 x i16> @v2regbug(ptr %B) nounwind {
;CHECK-LABEL: v2regbug:
;CHECK: vzip.16
	%tmp1 = load <4 x i16>, ptr %B
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32><i32 0, i32 0, i32 1, i32 1>
	ret <4 x i16> %tmp2
}
