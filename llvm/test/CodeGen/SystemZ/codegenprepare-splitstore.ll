; Test that CodeGenPrepare respects endianness when splitting a store.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -stop-after codegenprepare -force-split-store < %s  | FileCheck %s

define void @fun(i16* %Src, i16* %Dst) {
; CHECK-LABEL: @fun(
; CHECK:      %1 = load i16, i16* %Src
; CHECK-NEXT: %2 = trunc i16 %1 to i8
; CHECK-NEXT: %3 = lshr i16 %1, 8
; CHECK-NEXT: %4 = trunc i16 %3 to i8
; CHECK-NEXT: %5 = zext i8 %2 to i16
; CHECK-NEXT: %6 = zext i8 %4 to i16
; CHECK-NEXT: %7 = shl nuw i16 %6, 8
; CHECK-NEXT: %8 = or i16 %7, %5
; CHECK-NEXT: %9 = bitcast i16* %Dst to i8*
; CHECK-NEXT: %10 = getelementptr i8, i8* %9, i32 1
; CHECK-NEXT: store i8 %2, i8* %10
; CHECK-NEXT: %11 = bitcast i16* %Dst to i8*
; CHECK-NEXT: store i8 %4, i8* %11
  %1 = load i16, i16* %Src
  %2 = trunc i16 %1 to i8
  %3 = lshr i16 %1, 8
  %4 = trunc i16 %3 to i8
  %5 = zext i8 %2 to i16
  %6 = zext i8 %4 to i16
  %7 = shl nuw i16 %6, 8
  %8 = or i16 %7, %5
  store i16 %8, i16* %Dst
  ret void
}
