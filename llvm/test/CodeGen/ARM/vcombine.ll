; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc -mtriple=armeb-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE

define <16 x i8> @vcombine8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
; CHECK-LABEL: vcombine8
; CHECK-DAG: vldr [[LD0:d[0-9]+]], [r0]
; CHECK-DAG: vldr [[LD1:d[0-9]+]], [r1]

; CHECK-LE-DAG: vmov r0, r1, [[LD0]]
; CHECK-LE-DAG: vmov r2, r3, [[LD1]]

; CHECK-BE-DAG: vmov r1, r0, d16
; CHECK-BE-DAG: vmov r3, r2, d17
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
	ret <16 x i8> %tmp3
}

define <8 x i16> @vcombine16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
; CHECK-LABEL: vcombine16
; CHECK-DAG: vldr [[LD0:d[0-9]+]], [r0]
; CHECK-DAG: vldr [[LD1:d[0-9]+]], [r1]

; CHECK-LE-DAG: vmov r0, r1, [[LD0]]
; CHECK-LE-DAG: vmov r2, r3, [[LD1]]

; CHECK-BE-DAG: vmov r1, r0, d16
; CHECK-BE-DAG: vmov r3, r2, d17
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
	ret <8 x i16> %tmp3
}

define <4 x i32> @vcombine32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
; CHECK-LABEL: vcombine32

; CHECK-DAG: vldr [[LD0:d[0-9]+]], [r0]
; CHECK-DAG: vldr [[LD1:d[0-9]+]], [r1]

; CHECK-LE: vmov r0, r1, [[LD0]]
; CHECK-LE: vmov r2, r3, [[LD1]]

; CHECK-BE: vmov r1, r0, d16
; CHECK-BE: vmov r3, r2, d17
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
	ret <4 x i32> %tmp3
}

define <4 x float> @vcombinefloat(<2 x float>* %A, <2 x float>* %B) nounwind {
; CHECK-LABEL: vcombinefloat

; CHECK-DAG: vldr [[LD0:d[0-9]+]], [r0]
; CHECK-DAG: vldr [[LD1:d[0-9]+]], [r1]

; CHECK-LE: vmov r0, r1, [[LD0]]
; CHECK-LE: vmov r2, r3, [[LD1]]

; CHECK-BE: vmov r1, r0, d16
; CHECK-BE: vmov r3, r2, d17
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
	ret <4 x float> %tmp3
}

define <2 x i64> @vcombine64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
; CHECK-LABEL: vcombine64
; CHECK-DAG: vldr [[LD0:d[0-9]+]], [r0]
; CHECK-DAG: vldr [[LD1:d[0-9]+]], [r1]

; CHECK-LE: vmov r0, r1, [[LD0]]
; CHECK-LE: vmov r2, r3, [[LD1]]

; CHECK-BE: vmov r1, r0, [[LD0]]
; CHECK-BE: vmov r3, r2, [[LD1]]
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = shufflevector <1 x i64> %tmp1, <1 x i64> %tmp2, <2 x i32> <i32 0, i32 1>
	ret <2 x i64> %tmp3
}

; Check for vget_low and vget_high implemented with shufflevector.  PR8411.
; They should not require storing to the stack.

define <4 x i16> @vget_low16(<8 x i16>* %A) nounwind {
; CHECK: vget_low16
; CHECK-NOT: vst
; CHECK-LE: vmov r0, r1, d16
; CHECK-BE: vmov r1, r0, d16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i16> %tmp2
}

define <8 x i8> @vget_high8(<16 x i8>* %A) nounwind {
; CHECK: vget_high8
; CHECK-NOT: vst
; CHECK-LE: vmov r0, r1, d17
; CHECK-BE: vmov r1, r0, d16
	%tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <8 x i8> %tmp2
}

; vcombine(vld1_dup(p), vld1_dup(p2))
define <8 x i16> @vcombine_vdup(<8 x i16> %src, i16* nocapture readonly %p) {
; CHECK-LABEL: vcombine_vdup:
; CHECK: vld1.16 {d16[]},
; CHECK: vld1.16 {d17[]},
; CHECK-LE: vmov    r0, r1, d16
; CHECK-LE: vmov    r2, r3, d17
  %a1 = load i16, i16* %p, align 2
  %a2 = insertelement <4 x i16> undef, i16 %a1, i32 0
  %a3 = shufflevector <4 x i16> %a2, <4 x i16> undef, <4 x i32> zeroinitializer
  %p2 = getelementptr inbounds i16, i16* %p, i32 1
  %b1 = load i16, i16* %p2, align 2
  %b2 = insertelement <4 x i16> undef, i16 %b1, i32 0
  %b3 = shufflevector <4 x i16> %b2, <4 x i16> undef, <4 x i32> zeroinitializer
  %shuffle = shufflevector <4 x i16> %a3, <4 x i16> %b3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle
}
