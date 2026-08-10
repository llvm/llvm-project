; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

; <256 x i32>

; Function Attrs: nounwind
define fastcc <256 x i32> @add_vv_v256i32(<256 x i32> %x, <256 x i32> %y) {
; CHECK-LABEL: add_vv_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vadds.w.sx %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %z = add <256 x i32> %x, %y
  ret <256 x i32> %z
}

; Function Attrs: nounwind
define fastcc <256 x i32> @add_sv_v256i32(i32 %x, <256 x i32> %y) {
; CHECK-LABEL: add_sv_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vadds.w.sx %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %xins = insertelement <256 x i32> undef, i32 %x, i32 0
  %vx = shufflevector <256 x i32> %xins, <256 x i32> undef, <256 x i32> zeroinitializer
  %z = add <256 x i32> %vx, %y
  ret <256 x i32> %z
}

; Function Attrs: nounwind
define fastcc <256 x i32> @add_vs_v256i32(<256 x i32> %x, i32 %y) {
; CHECK-LABEL: add_vs_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vadds.w.sx %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %yins = insertelement <256 x i32> undef, i32 %y, i32 0
  %vy = shufflevector <256 x i32> %yins, <256 x i32> undef, <256 x i32> zeroinitializer
  %z = add <256 x i32> %x, %vy
  ret <256 x i32> %z
}



; <256 x i64>

; Function Attrs: nounwind
define fastcc <256 x i64> @add_vv_v256i64(<256 x i64> %x, <256 x i64> %y) {
; CHECK-LABEL: add_vv_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vadds.l %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %z = add <256 x i64> %x, %y
  ret <256 x i64> %z
}

; Function Attrs: nounwind
define fastcc <256 x i64> @add_sv_v256i64(i64 %x, <256 x i64> %y) {
; CHECK-LABEL: add_sv_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vadds.l %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %xins = insertelement <256 x i64> undef, i64 %x, i32 0
  %vx = shufflevector <256 x i64> %xins, <256 x i64> undef, <256 x i32> zeroinitializer
  %z = add <256 x i64> %vx, %y
  ret <256 x i64> %z
}

; Function Attrs: nounwind
define fastcc <256 x i64> @add_vs_v256i64(<256 x i64> %x, i64 %y) {
; CHECK-LABEL: add_vs_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vadds.l %v0, %s0, %v0
; CHECK-NEXT:    b.l.t (, %s10)
  %yins = insertelement <256 x i64> undef, i64 %y, i32 0
  %vy = shufflevector <256 x i64> %yins, <256 x i64> undef, <256 x i32> zeroinitializer
  %z = add <256 x i64> %x, %vy
  ret <256 x i64> %z
}

; <128 x i64>
; We expect this to be widened.

; Function Attrs: nounwind
define fastcc <128 x i64> @add_vv_v128i64(<128 x i64> %x, <128 x i64> %y) {
; CHECK-LABEL: add_vv_v128i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vadds.l %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %z = add <128 x i64> %x, %y
  ret <128 x i64> %z
}

; <256 x i16>
; We expect promotion.

; Function Attrs: nounwind
define fastcc <256 x i16> @add_vv_v256i16(<256 x i16> %x, <256 x i16> %y) {
; CHECK-LABEL: add_vv_v256i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vadds.w.sx %v0, %v0, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  %z = add <256 x i16> %x, %y
  ret <256 x i16> %z
}

; <128 x i16>
; We expect this to be scalarized (for now).

; Function Attrs: nounwind
define fastcc <128 x i16> @add_vv_v128i16(<128 x i16> %x, <128 x i16> %y) {
; CHECK-LABEL: add_vv_v128i16:
; CHECK-NOT:     vadd
  %z = add <128 x i16> %x, %y
  ret <128 x i16> %z
}

