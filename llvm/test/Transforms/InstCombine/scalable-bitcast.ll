; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; We shouldn't fold bitcast(insert <vscale x 1 x iX> .., iX %val, i32 0)
; into bitcast(iX %val) for scalable vectors.
define <vscale x 2 x i8> @bitcast_of_insert_i8_i16(i16 %val) #0 {
; CHECK-LABEL: @bitcast_of_insert_i8_i16(
; CHECK-NOT:   bitcast i16 %val to <vscale x 2 x i8>
; CHECK:       bitcast <vscale x 1 x i16> %op2 to <vscale x 2 x i8>
entry:
  %op2 = insertelement <vscale x 1 x i16> undef, i16 %val, i32 0
  %0 = bitcast <vscale x 1 x i16> %op2 to <vscale x 2 x i8>
  ret <vscale x 2 x i8> %0
}
