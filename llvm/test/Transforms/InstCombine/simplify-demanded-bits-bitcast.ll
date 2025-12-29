; RUN: opt -passes=instcombine -S %s | FileCheck %s

; this test verifies that SimplifyDemandedBits can peek through
; lossless bitcasts with identical scalar bit widths.

define i32 @bitcast_signbit_only(i32 %x) {
; CHECK-LABEL: @bitcast_signbit_only(
; CHECK-NEXT: ret i32 0

  ; Clear the sign bit explicitly
  %masked = and i32 %x, 2147483647   ; 0x7fffffff

  ; Bitcast i32 -> float -> i32
  %f = bitcast i32 %masked to float
  %bc = bitcast float %f to i32

  ; Use only the sign bit
  %sign = ashr i32 %bc, 31
  ret i32 %sign
}
