; RUN: llc < %s | FileCheck %s
target triple = "arm64-apple-ios7.0"

define i64 @foo(i64* nocapture readonly %ptr, i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
; CHECK: ldr     w8, [x0, #4]
; CHECK: lsr     x9, x1, #32
; CHECK: cmn             x3, x2
; CHECK: mul             x8, x8, x9
; CHECK: cinc     x0, x8, hs
; CHECK: ret
entry:
  %0 = lshr i64 %a, 32
  %1 = load i64, i64* %ptr, align 8
  %2 = lshr i64 %1, 32
  %3 = mul nuw i64 %2, %0
  %4 = add i64 %c, %b
  %5 = icmp ult i64 %4, %c
  %6 = zext i1 %5 to i64
  %7 = add i64 %3, %6
  ret i64 %7
}

attributes #0 = { norecurse nounwind readonly }
