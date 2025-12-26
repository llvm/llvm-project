; This is the DirectX counterpart to the generic `half.ll` test as DirectX has compilation errors
; on some operations.
; RUN: llc %s -o - -mtriple=dxil-pc-shadermodel6.3-library | FileCheck %s

define half @from_bits(i16 %bits) nounwind {
; CHECK-LABEL: @from_bits
; CHECK: %f = bitcast i16 %bits to half
; CHECK-NEXT: ret half %f
  %f = bitcast i16 %bits to half
  ret half %f
}

define i16 @to_bits(half %f) nounwind {
; CHECK-LABEL: @to_bits
; CHECK: %bits = bitcast half %f to i16
; CHECK-NEXT: ret i16 %bits
  %bits = bitcast half %f to i16
  ret i16 %bits
}

define half @check_freeze(half %f) nounwind {
; CHECK-LABEL: @check_freeze
; CHECK: ret half %f
  %t0 = freeze half %f
  ret half %t0
}
