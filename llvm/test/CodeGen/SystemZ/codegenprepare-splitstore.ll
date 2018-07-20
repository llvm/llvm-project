; Test that CodeGenPrepare respects endianness when splitting a store.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -force-split-store < %s  | FileCheck %s

define void @fun(i16* %Src, i16* %Dst) {
; CHECK-LABEL: # %bb.0:
; CHECK:       lh   %r0, 0(%r2)
; CHECK-NEXT:  stc  %r0, 1(%r3)
; CHECK-NEXT:  srl  %r0, 8
; CHECK-NEXT:  stc  %r0, 0(%r3)
; CHECK-NEXT:  br   %r14
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
