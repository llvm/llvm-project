; Test that CodeGenPrepare respects endianness when splitting a store.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -stop-after codegenprepare -force-split-store < %s  | FileCheck %s

define void @fun(ptr %Src, ptr %Dst) {
; CHECK-LABEL: @fun(
; CHECK:      %1 = load i16, ptr %Src
; CHECK-NEXT: %2 = trunc i16 %1 to i8
; CHECK-NEXT: %3 = lshr i16 %1, 8
; CHECK-NEXT: %4 = trunc i16 %3 to i8
; CHECK-NEXT: %5 = zext i8 %2 to i16
; CHECK-NEXT: %6 = zext i8 %4 to i16
; CHECK-NEXT: %7 = shl nuw i16 %6, 8
; CHECK-NEXT: %8 = or i16 %7, %5
; CHECK-NEXT: %9 = getelementptr i8, ptr %Dst, i32 1
; CHECK-NEXT: store i8 %2, ptr %9
; CHECK-NEXT: store i8 %4, ptr %Dst
  %1 = load i16, ptr %Src
  %2 = trunc i16 %1 to i8
  %3 = lshr i16 %1, 8
  %4 = trunc i16 %3 to i8
  %5 = zext i8 %2 to i16
  %6 = zext i8 %4 to i16
  %7 = shl nuw i16 %6, 8
  %8 = or i16 %7, %5
  store i16 %8, ptr %Dst
  ret void
}
