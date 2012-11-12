; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; This tests icmp operations that do not map directly to NEON instructions.
; Not-equal (ne) operations are implemented by VCEQ/VMVN.  Less-than (lt/ult)
; and less-than-or-equal (le/ule) are implemented by swapping the arguments
; to VCGT and VCGE.  Test all the operand types for not-equal but only sample
; the other operations.

define <8 x i8> @vcnei8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcnei8:
;CHECK: vceq.i8
;CHECK-NEXT: vmvn
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp ne <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcnei16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcnei16:
;CHECK: vceq.i16
;CHECK-NEXT: vmvn
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp ne <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcnei32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcnei32:
;CHECK: vceq.i32
;CHECK-NEXT: vmvn
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp ne <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vcneQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcneQi8:
;CHECK: vceq.i8
;CHECK-NEXT: vmvn
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp ne <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcneQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcneQi16:
;CHECK: vceq.i16
;CHECK-NEXT: vmvn
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp ne <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcneQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcneQi32:
;CHECK: vceq.i32
;CHECK-NEXT: vmvn
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp ne <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <16 x i8> @vcltQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcltQs8:
;CHECK: vcgt.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp slt <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @vcles16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcles16:
;CHECK: vcge.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp sle <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <4 x i16> @vcltu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcltu16:
;CHECK: vcgt.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp ult <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <4 x i32> @vcleQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcleQu32:
;CHECK: vcge.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp ule <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
