; RUN: opt -S -instsimplify < %s | FileCheck %s

; CHECK-LABEL: definitely_in_bounds
; CHECK: ret i8 0
define i8 @definitely_in_bounds() {
  ret i8 extractelement (<vscale x 16 x i8> zeroinitializer, i64 15)
}

; CHECK-LABEL: maybe_in_bounds
; CHECK: ret i8 extractelement (<vscale x 16 x i8> zeroinitializer, i64 16)
define i8 @maybe_in_bounds() {
  ret i8 extractelement (<vscale x 16 x i8> zeroinitializer, i64 16)
}

; Examples of extracting a lane from a splat constant

define i32 @extractconstant_shuffle_in_range(i32 %v) {
; CHECK-LABEL: @extractconstant_shuffle_in_range(
; CHECK-NEXT:    ret i32 1024
;
  %in = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %in, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %r = extractelement <vscale x 4 x i32> %splat, i32 1
  ret i32 %r
}

define i32 @extractconstant_shuffle_maybe_out_of_range(i32 %v) {
; CHECK-LABEL: @extractconstant_shuffle_maybe_out_of_range(
; CHECK-NEXT:    ret i32 extractelement (<vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 1024, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer), i32 4)
;
  %in = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %in, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %r = extractelement <vscale x 4 x i32> %splat, i32 4
  ret i32 %r
}

define i32 @extractconstant_shuffle_invalid_index(i32 %v) {
; CHECK-LABEL: @extractconstant_shuffle_invalid_index(
; CHECK-NEXT:    ret i32 extractelement (<vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 1024, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer), i32 -1)
;
  %in = insertelement <vscale x 4 x i32> undef, i32 1024, i32 0
  %splat = shufflevector <vscale x 4 x i32> %in, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %r = extractelement <vscale x 4 x i32> %splat, i32 -1
  ret i32 %r
}
