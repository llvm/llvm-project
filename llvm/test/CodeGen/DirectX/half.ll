; This is the DirectX counterpart to the generic `half.ll` test as DirectX has compilation errors
; on some operations.
; RUN: llc %s -o - -mtriple=dxil-pc-shadermodel6.3-library | FileCheck %s

; As this is a graphics target, this just checks that compilation doesn't crash.
; CHECK: {{.*}}

define half @from_bits(i16 %bits) nounwind {
    %f = bitcast i16 %bits to half
    ret half %f
}

define i16 @to_bits(half %f) nounwind {
    %bits = bitcast half %f to i16
    ret i16 %bits
}

define half @check_freeze(half %f) nounwind {
  %t0 = freeze half %f
  ret half %t0
}
