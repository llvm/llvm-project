; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i16> @vshlls8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vshlls8:
;CHECK: vshll.s8
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %sext = sext <8 x i8> %tmp1 to <8 x i16>
        %shift = shl <8 x i16> %sext, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
        ret <8 x i16> %shift
}

define <4 x i32> @vshlls16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vshlls16:
;CHECK: vshll.s16
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %sext = sext <4 x i16> %tmp1 to <4 x i32>
        %shift = shl <4 x i32> %sext, <i32 15, i32 15, i32 15, i32 15>
        ret <4 x i32> %shift
}

define <2 x i64> @vshlls32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vshlls32:
;CHECK: vshll.s32
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %sext = sext <2 x i32> %tmp1 to <2 x i64>
        %shift = shl <2 x i64> %sext, <i64 31, i64 31>
        ret <2 x i64> %shift
}

define <8 x i16> @vshllu8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vshllu8:
;CHECK: vshll.u8
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %zext = zext <8 x i8> %tmp1 to <8 x i16>
        %shift = shl <8 x i16> %zext, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
        ret <8 x i16> %shift
}

define <4 x i32> @vshllu16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vshllu16:
;CHECK: vshll.u16
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %zext = zext <4 x i16> %tmp1 to <4 x i32>
        %shift = shl <4 x i32> %zext, <i32 15, i32 15, i32 15, i32 15>
        ret <4 x i32> %shift
}

define <2 x i64> @vshllu32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vshllu32:
;CHECK: vshll.u32
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %zext = zext <2 x i32> %tmp1 to <2 x i64>
        %shift = shl <2 x i64> %zext, <i64 31, i64 31>
        ret <2 x i64> %shift
}

; The following tests use the maximum shift count, so the signedness is
; irrelevant.  Test both signed and unsigned versions.
define <8 x i16> @vshlli8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vshlli8:
;CHECK: vshll.i8
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %sext = sext <8 x i8> %tmp1 to <8 x i16>
        %shift = shl <8 x i16> %sext, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
        ret <8 x i16> %shift
}

define <4 x i32> @vshlli16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vshlli16:
;CHECK: vshll.i16
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %zext = zext <4 x i16> %tmp1 to <4 x i32>
        %shift = shl <4 x i32> %zext, <i32 16, i32 16, i32 16, i32 16>
        ret <4 x i32> %shift
}

define <2 x i64> @vshlli32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vshlli32:
;CHECK: vshll.i32
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %zext = zext <2 x i32> %tmp1 to <2 x i64>
        %shift = shl <2 x i64> %zext, <i64 32, i64 32>
        ret <2 x i64> %shift
}

; And these have a shift just out of range so separate vmovl and vshl
; instructions are needed.
define <8 x i16> @vshllu8_bad(<8 x i8>* %A) nounwind {
; CHECK-LABEL: vshllu8_bad:
; CHECK: vmovl.u8
; CHECK: vshl.i16
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %zext = zext <8 x i8> %tmp1 to <8 x i16>
        %shift = shl <8 x i16> %zext, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
        ret <8 x i16> %shift
}

define <4 x i32> @vshlls16_bad(<4 x i16>* %A) nounwind {
; CHECK-LABEL: vshlls16_bad:
; CHECK: vmovl.u16
; CHECK: vshl.i32
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %sext = sext <4 x i16> %tmp1 to <4 x i32>
        %shift = shl <4 x i32> %sext, <i32 17, i32 17, i32 17, i32 17>
        ret <4 x i32> %shift
}

define <2 x i64> @vshllu32_bad(<2 x i32>* %A) nounwind {
; CHECK-LABEL: vshllu32_bad:
; CHECK: vmovl.u32
; CHECK: vshl.i64
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %zext = zext <2 x i32> %tmp1 to <2 x i64>
        %shift = shl <2 x i64> %zext, <i64 33, i64 33>
        ret <2 x i64> %shift
}
