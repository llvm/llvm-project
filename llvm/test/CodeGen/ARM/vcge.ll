; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vcges8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcges8:
;CHECK: vcge.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp sge <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcges16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcges16:
;CHECK: vcge.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp sge <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcges32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcges32:
;CHECK: vcge.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp sge <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <8 x i8> @vcgeu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcgeu8:
;CHECK: vcge.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp uge <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcgeu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcgeu16:
;CHECK: vcge.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp uge <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcgeu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcgeu32:
;CHECK: vcge.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp uge <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <2 x i32> @vcgef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vcgef32:
;CHECK: vcge.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fcmp oge <2 x float> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vcgeQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcgeQs8:
;CHECK: vcge.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp sge <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgeQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcgeQs16:
;CHECK: vcge.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp sge <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgeQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcgeQs32:
;CHECK: vcge.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp sge <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <16 x i8> @vcgeQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcgeQu8:
;CHECK: vcge.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp uge <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgeQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcgeQu16:
;CHECK: vcge.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp uge <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgeQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcgeQu32:
;CHECK: vcge.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp uge <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <4 x i32> @vcgeQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vcgeQf32:
;CHECK: vcge.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fcmp oge <4 x float> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i32> @vacgef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vacgef32:
;CHECK: vacge.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vacged(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @vacgeQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vacgeQf32:
;CHECK: vacge.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vacgeq(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

declare <2 x i32> @llvm.arm.neon.vacged(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vacgeq(<4 x float>, <4 x float>) nounwind readnone

define <8 x i8> @vcgei8Z(<8 x i8>* %A) nounwind {
;CHECK: vcgei8Z:
;CHECK-NOT: vmov
;CHECK-NOT: vmvn
;CHECK: vcge.s8
	%tmp1 = load <8 x i8>* %A
	%tmp3 = icmp sge <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <8 x i8> @vclei8Z(<8 x i8>* %A) nounwind {
;CHECK: vclei8Z:
;CHECK-NOT: vmov
;CHECK-NOT: vmvn
;CHECK: vcle.s8
	%tmp1 = load <8 x i8>* %A
	%tmp3 = icmp sle <8 x i8> %tmp1, <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; Radar 8782191
; Floating-point comparisons against zero produce results with integer
; elements, not floating-point elements.
define void @test_vclez_fp() nounwind optsize {
;CHECK: test_vclez_fp
;CHECK: vcle.f32
entry:
  %0 = fcmp ole <4 x float> undef, zeroinitializer
  %1 = sext <4 x i1> %0 to <4 x i16>
  %2 = add <4 x i16> %1, zeroinitializer
  %3 = shufflevector <4 x i16> %2, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %4 = add <8 x i16> %3, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %5 = trunc <8 x i16> %4 to <8 x i8>
  tail call void @llvm.arm.neon.vst1.v8i8(i8* undef, <8 x i8> %5, i32 1)
  unreachable
}

declare void @llvm.arm.neon.vst1.v8i8(i8*, <8 x i8>, i32) nounwind
