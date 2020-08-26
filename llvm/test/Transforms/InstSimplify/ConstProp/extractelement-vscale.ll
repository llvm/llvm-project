; RUN: opt -instcombine -S < %s | FileCheck %s

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
