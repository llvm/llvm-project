; RUN: opt -passes=instcombine -S %s | FileCheck %s

define i1 @demanded_signbit_v1i32_bitcast(<1 x i32> %x) {
; CHECK-LABEL: @demanded_signbit_v1i32_bitcast(
; CHECK: extractelement <1 x i32>
; CHECK: icmp slt i32
; CHECK: ret i1

  ; Single bitcast: v1i32 -> i32
  %bc = bitcast <1 x i32> %x to i32

  ; Only the sign bit is demanded here
  %cmp = icmp slt i32 %bc, 0
  ret i1 %cmp
}
