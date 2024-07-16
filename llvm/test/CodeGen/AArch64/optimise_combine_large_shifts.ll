; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=neon | FileCheck %s

define dso_local <16 x i8> @combine16ix8(<8 x i16> noundef %0, <8 x i16> noundef %1) local_unnamed_addr #0 {
; CHECK-LABEL: combine16ix8
; CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: ushr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #2
; CHECK-NEXT: ret
  %3 = lshr <8 x i16> %0, <i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10>
  %4 = lshr <8 x i16> %1, <i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10>
  %5 = shufflevector <8 x i16> %3, <8 x i16> %4, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %6 = trunc nuw nsw <16 x i16> %5 to <16 x i8>
  ret <16 x i8> %6
}

define dso_local <8 x i16> @combine32ix4(<4 x i32> noundef %0, <4 x i32> noundef %1) local_unnamed_addr #0 {
; CHECK-LABEL: combine32ix4
; CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK-NEXT: ushr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #4
; CHECK-NEXT: ret
  %3 = lshr <4 x i32> %0, <i32 20, i32 20, i32 20, i32 20>
  %4 = lshr <4 x i32> %1, <i32 20, i32 20, i32 20, i32 20>
  %5 = shufflevector <4 x i32> %3, <4 x i32> %4, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %6 = trunc nuw nsw <8 x i32> %5 to <8 x i16>
  ret <8 x i16> %6
}

define dso_local <4 x i32> @combine64ix2(<2 x i64> noundef %0, <2 x i64> noundef %1) local_unnamed_addr #0 {
; CHECK-LABEL: combine64ix2
; CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: ushr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #8
; CHECK-NEXT: ret
  %3 = lshr <2 x i64> %0, <i64 40, i64 40>
  %4 = lshr <2 x i64> %1, <i64 40, i64 40>
  %5 = shufflevector <2 x i64> %3, <2 x i64> %4, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = trunc nuw nsw <4 x i64> %5 to <4 x i32>
  ret <4 x i32> %6
}