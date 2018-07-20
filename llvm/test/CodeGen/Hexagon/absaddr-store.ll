; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s
; Check that we generate load instructions with absolute addressing mode.

@a0 = external global i32
@a1 = external global i32
@b0 = external global i8
@b1 = external global i8
@c0 = external global i16
@c1 = external global i16
@d = external global i64

define zeroext i8 @absStoreByte() nounwind {
; CHECK: memb(##b1) = r{{[0-9]+}}
entry:
  %0 = load i8, i8* @b0, align 1
  %conv = zext i8 %0 to i32
  %mul = mul nsw i32 100, %conv
  %conv1 = trunc i32 %mul to i8
  store i8 %conv1, i8* @b1, align 1
  ret i8 %conv1
}

define signext i16 @absStoreHalf() nounwind {
; CHECK: memh(##c1) = r{{[0-9]+}}
entry:
  %0 = load i16, i16* @c0, align 2
  %conv = sext i16 %0 to i32
  %mul = mul nsw i32 100, %conv
  %conv1 = trunc i32 %mul to i16
  store i16 %conv1, i16* @c1, align 2
  ret i16 %conv1
}

define i32 @absStoreWord() nounwind {
; CHECK: memw(##a1) = r{{[0-9]+}}
entry:
  %0 = load i32, i32* @a0, align 4
  %mul = mul nsw i32 100, %0
  store i32 %mul, i32* @a1, align 4
  ret i32 %mul
}

define void @absStoreDouble() nounwind {
; CHECK: memd(##d) = r{{[0-9]+}}:{{[0-9]+}}
entry:
  store i64 100, i64* @d, align 8
  ret void
}

