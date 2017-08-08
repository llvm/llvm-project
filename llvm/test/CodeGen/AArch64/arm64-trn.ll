; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @vtrni8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vtrni8:
;CHECK: trn1.8b
;CHECK: trn2.8b
;CHECK-NEXT: add.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
	%tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vtrni16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vtrni16:
;CHECK: trn1.4h
;CHECK: trn2.4h
;CHECK-NEXT: add.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
	%tmp4 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
        %tmp5 = add <4 x i16> %tmp3, %tmp4
	ret <4 x i16> %tmp5
}

; 2xi32 TRN is redundant with ZIP
define <2 x i32> @vtrni32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vtrni32:
;CHECK: zip1.2s
;CHECK: zip2.2s
;CHECK-NEXT: add.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <2 x i32> <i32 0, i32 2>
	%tmp4 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <2 x i32> <i32 1, i32 3>
        %tmp5 = add <2 x i32> %tmp3, %tmp4
	ret <2 x i32> %tmp5
}

define <2 x float> @vtrnf(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vtrnf:
;CHECK: zip1.2s
;CHECK: zip2.2s
;CHECK-NEXT: fadd.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <2 x i32> <i32 0, i32 2>
	%tmp4 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <2 x i32> <i32 1, i32 3>
        %tmp5 = fadd <2 x float> %tmp3, %tmp4
	ret <2 x float> %tmp5
}

define <16 x i8> @vtrnQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vtrnQi8:
;CHECK: trn1.16b
;CHECK: trn2.16b
;CHECK-NEXT: add.16b
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
	%tmp4 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
        %tmp5 = add <16 x i8> %tmp3, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vtrnQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vtrnQi16:
;CHECK: trn1.8h
;CHECK: trn2.8h
;CHECK-NEXT: add.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
	%tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vtrnQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vtrnQi32:
;CHECK: trn1.4s
;CHECK: trn2.4s
;CHECK-NEXT: add.4s
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
	%tmp4 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
        %tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vtrnQf(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vtrnQf:
;CHECK: trn1.4s
;CHECK: trn2.4s
;CHECK-NEXT: fadd.4s
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
	%tmp4 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
        %tmp5 = fadd <4 x float> %tmp3, %tmp4
	ret <4 x float> %tmp5
}

; Undef shuffle indices should not prevent matching to VTRN:

define <8 x i8> @vtrni8_undef(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vtrni8_undef:
;CHECK: trn1.8b
;CHECK: trn2.8b
;CHECK-NEXT: add.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 undef, i32 2, i32 10, i32 undef, i32 12, i32 6, i32 14>
	%tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 undef, i32 undef, i32 15>
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <8 x i16> @vtrnQi16_undef(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vtrnQi16_undef:
;CHECK: trn1.8h
;CHECK: trn2.8h
;CHECK-NEXT: add.8h
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 8, i32 undef, i32 undef, i32 4, i32 12, i32 6, i32 14>
	%tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 1, i32 undef, i32 3, i32 11, i32 5, i32 13, i32 undef, i32 undef>
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}
