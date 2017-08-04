; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; RUN: llc -mtriple=arm-eabi -mattr=+neon -regalloc=basic %s -o - | FileCheck %s

define <8 x i8> @vcgts8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vcgts8:
;CHECK: vcgt.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = icmp sgt <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcgts16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vcgts16:
;CHECK: vcgt.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = icmp sgt <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcgts32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vcgts32:
;CHECK: vcgt.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = icmp sgt <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <8 x i8> @vcgtu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vcgtu8:
;CHECK: vcgt.u8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = icmp ugt <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcgtu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vcgtu16:
;CHECK: vcgt.u16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = icmp ugt <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcgtu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vcgtu32:
;CHECK: vcgt.u32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = icmp ugt <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <2 x i32> @vcgtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vcgtf32:
;CHECK: vcgt.f32
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = fcmp ogt <2 x float> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vcgtQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vcgtQs8:
;CHECK: vcgt.s8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = icmp sgt <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgtQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vcgtQs16:
;CHECK: vcgt.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = icmp sgt <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgtQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vcgtQs32:
;CHECK: vcgt.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = icmp sgt <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <16 x i8> @vcgtQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vcgtQu8:
;CHECK: vcgt.u8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = icmp ugt <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgtQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vcgtQu16:
;CHECK: vcgt.u16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = icmp ugt <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgtQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vcgtQu32:
;CHECK: vcgt.u32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = icmp ugt <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <4 x i32> @vcgtQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vcgtQf32:
;CHECK: vcgt.f32
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = fcmp ogt <4 x float> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i32> @vacgtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vacgtf32:
;CHECK: vacgt.f32
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vacgt.v2i32.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @vacgtQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vacgtQf32:
;CHECK: vacgt.f32
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vacgt.v4i32.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

; rdar://7923010
define <4 x i32> @vcgt_zext(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vcgt_zext:
;CHECK-DAG: vmov.i32 [[Q0:q[0-9]+]], #0x1
;CHECK-DAG: vcgt.f32 [[Q1:q[0-9]+]]
;CHECK: vand [[Q2:q[0-9]+]], [[Q1]], [[Q0]]
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = fcmp ogt <4 x float> %tmp1, %tmp2
        %tmp4 = zext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

declare <2 x i32> @llvm.arm.neon.vacgt.v2i32.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vacgt.v4i32.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <8 x i8> @vcgti8Z(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vcgti8Z:
;CHECK-NOT: vmov
;CHECK-NOT: vmvn
;CHECK: vcgt.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp3 = icmp sgt <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <8 x i8> @vclti8Z(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vclti8Z:
;CHECK-NOT: vmov
;CHECK-NOT: vmvn
;CHECK: vclt.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp3 = icmp slt <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}
