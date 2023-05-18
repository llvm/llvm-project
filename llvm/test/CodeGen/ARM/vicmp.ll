; RUN: llc -mtriple=arm -mattr=+neon %s -o - | FileCheck %s

; This tests icmp operations that do not map directly to NEON instructions.
; Not-equal (ne) operations are implemented by VCEQ/VMVN.  Less-than (lt/ult)
; and less-than-or-equal (le/ule) are implemented by swapping the arguments
; to VCGT and VCGE.  Test all the operand types for not-equal but only sample
; the other operations.

define <8 x i8> @vcnei8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcnei8:
;CHECK: vceq.i8
;CHECK-NEXT: vmvn
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = icmp ne <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcnei16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcnei16:
;CHECK: vceq.i16
;CHECK-NEXT: vmvn
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = icmp ne <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcnei32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcnei32:
;CHECK: vceq.i32
;CHECK-NEXT: vmvn
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = icmp ne <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vcneQi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcneQi8:
;CHECK: vceq.i8
;CHECK-NEXT: vmvn
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = icmp ne <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcneQi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcneQi16:
;CHECK: vceq.i16
;CHECK-NEXT: vmvn
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = icmp ne <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcneQi32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcneQi32:
;CHECK: vceq.i32
;CHECK-NEXT: vmvn
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = icmp ne <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <16 x i8> @vcltQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcltQs8:
;CHECK: vcgt.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = icmp slt <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @vcles16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcles16:
;CHECK: vcge.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = icmp sle <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <4 x i16> @vcltu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcltu16:
;CHECK: vcgt.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = icmp ult <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <4 x i32> @vcleQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vcleQu32:
;CHECK: vcge.u32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = icmp ule <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
