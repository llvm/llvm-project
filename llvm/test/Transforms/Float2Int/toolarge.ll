; RUN: opt < %s -passes=float2int -float2int-max-integer-bw=256 -S | FileCheck %s

; CHECK-LABEL: @neg_toolarge
; CHECK:  %1 = uitofp i80 %a to fp128
; CHECK:  %2 = fadd fp128 %1, %1
; CHECK:  %3 = fptoui fp128 %2 to i80
; CHECK:  ret i80 %3
; fp128 has a 112-bit mantissa, which can hold an i80. But we only support
; up to i64, so it should fail (even though the max integer bitwidth is 256).
define i80 @neg_toolarge(i80 %a) {
  %1 = uitofp i80 %a to fp128
  %2 = fadd fp128 %1, %1
  %3 = fptoui fp128 %2 to i80
  ret i80 %3
}

