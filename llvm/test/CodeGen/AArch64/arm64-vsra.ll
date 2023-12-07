; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @vsras8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsras8:
;CHECK: ssra.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = ashr <8 x i8> %tmp2, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vsras16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsras16:
;CHECK: ssra.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = ashr <4 x i16> %tmp2, < i16 15, i16 15, i16 15, i16 15 >
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vsras32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsras32:
;CHECK: ssra.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = ashr <2 x i32> %tmp2, < i32 31, i32 31 >
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <16 x i8> @vsraQs8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQs8:
;CHECK: ssra.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = ashr <16 x i8> %tmp2, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vsraQs16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQs16:
;CHECK: ssra.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = ashr <8 x i16> %tmp2, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsraQs32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQs32:
;CHECK: ssra.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = ashr <4 x i32> %tmp2, < i32 31, i32 31, i32 31, i32 31 >
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsraQs64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQs64:
;CHECK: ssra.2d
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = ashr <2 x i64> %tmp2, < i64 63, i64 63 >
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i8> @vsrau8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsrau8:
;CHECK: usra.8b
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = lshr <8 x i8> %tmp2, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
        %tmp4 = add <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vsrau16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsrau16:
;CHECK: usra.4h
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = lshr <4 x i16> %tmp2, < i16 15, i16 15, i16 15, i16 15 >
        %tmp4 = add <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vsrau32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsrau32:
;CHECK: usra.2s
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = lshr <2 x i32> %tmp2, < i32 31, i32 31 >
        %tmp4 = add <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}


define <16 x i8> @vsraQu8(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQu8:
;CHECK: usra.16b
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = lshr <16 x i8> %tmp2, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
        %tmp4 = add <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vsraQu16(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQu16:
;CHECK: usra.8h
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = lshr <8 x i16> %tmp2, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
        %tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vsraQu32(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQu32:
;CHECK: usra.4s
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = lshr <4 x i32> %tmp2, < i32 31, i32 31, i32 31, i32 31 >
        %tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vsraQu64(ptr %A, ptr %B) nounwind {
;CHECK-LABEL: vsraQu64:
;CHECK: usra.2d
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = lshr <2 x i64> %tmp2, < i64 63, i64 63 >
        %tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <1 x i64> @vsra_v1i64(<1 x i64> %A, <1 x i64> %B) nounwind {
; CHECK-LABEL: vsra_v1i64:
; CHECK: ssra d0, d1, #63
  %tmp3 = ashr <1 x i64> %B, < i64 63 >
  %tmp4 = add <1 x i64> %A, %tmp3
  ret <1 x i64> %tmp4
}
