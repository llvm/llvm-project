; RUN: opt -passes=slp-vectorizer -S -slp-vectorize-non-power-of-2 < %s | FileCheck %s

define i64 @bswap_i24(ptr noalias %p, ptr noalias %p1) {
  %g1 = getelementptr i8, ptr %p, i32 1
  %g2 = getelementptr i8, ptr %p, i32 2

  %t0 = load i8, ptr %p
  %t1 = load i8, ptr %g1
  %t2 = load i8, ptr %g2

  %g11 = getelementptr i8, ptr %p1, i32 1
  %g12 = getelementptr i8, ptr %p1, i32 2

  %t10 = load i8, ptr %p1
  %t11 = load i8, ptr %g11
  %t12 = load i8, ptr %g12

  %a0 = add i8 %t0, %t10
  %a1 = add i8 %t1, %t11
  %a2 = add i8 %t2, %t12

  %z0 = zext i8 %a0 to i64
  %z1 = zext i8 %a1 to i64
  %z2 = zext i8 %a2 to i64

  %sh0 = shl nuw i64 %z0, 16
  %sh1 = shl nuw nsw i64 %z1, 8

  %or01 = or disjoint i64 %sh0, %sh1
  %or012 = or disjoint i64 %or01, %z2

  ret i64 %or012
}
