; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @vzipi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipi8:
;CHECK: zip1.8b
;CHECK: zip2.8b
;CHECK-NEXT: add.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
	%tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vzipi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipi16:
;CHECK: zip1.4h
;CHECK: zip2.4h
;CHECK-NEXT: add.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
	%tmp4 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
        %tmp5 = add <4 x i16> %tmp3, %tmp4
	ret <4 x i16> %tmp5
}

define <16 x i8> @vzipQi8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipQi8:
;CHECK: zip1.16b
;CHECK: zip2.16b
;CHECK-NEXT: add.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
	%tmp4 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
        %tmp5 = add <16 x i8> %tmp3, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vzipQi16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipQi16:
;CHECK: zip1.8h
;CHECK: zip2.8h
;CHECK-NEXT: add.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
	%tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vzipQi32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipQi32:
;CHECK: zip1.4s
;CHECK: zip2.4s
;CHECK-NEXT: add.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
	%tmp4 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
        %tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vzipQf(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipQf:
;CHECK: zip1.4s
;CHECK: zip2.4s
;CHECK-NEXT: fadd.4s
	%tmp1 = load <4 x float>, ptr %A
	%tmp2 = load <4 x float>, ptr %B
	%tmp3 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
	%tmp4 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
        %tmp5 = fadd <4 x float> %tmp3, %tmp4
	ret <4 x float> %tmp5
}

; Undef shuffle indices should not prevent matching to VZIP:

define <8 x i8> @vzipi8_undef(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipi8_undef:
;CHECK: zip1.8b
;CHECK: zip2.8b
;CHECK-NEXT: add.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 undef, i32 1, i32 9, i32 undef, i32 10, i32 3, i32 11>
	%tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 undef, i32 undef, i32 15>
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <16 x i8> @vzipQi8_undef(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vzipQi8_undef:
;CHECK: zip1.16b
;CHECK: zip2.16b
;CHECK-NEXT: add.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 16, i32 1, i32 undef, i32 undef, i32 undef, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
	%tmp4 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 8, i32 24, i32 9, i32 undef, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 undef, i32 14, i32 30, i32 undef, i32 31>
        %tmp5 = add <16 x i8> %tmp3, %tmp4
	ret <16 x i8> %tmp5
}
