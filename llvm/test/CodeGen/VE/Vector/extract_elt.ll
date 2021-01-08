; RUN: llc < %s -mtriple=ve-unknown-unknown -mattr=+vpu | FileCheck %s


;;; <256 x i64>

define fastcc i64 @extract_rr_v256i64(i32 signext %idx, <256 x i64> %v) {
; CHECK-LABEL: extract_rr_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x i64> %v, i32 %idx
  ret i64 %ret
}

define fastcc i64 @extract_ri7_v256i64(<256 x i64> %v) {
; CHECK-LABEL: extract_ri7_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(127)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x i64> %v, i32 127
  ret i64 %ret
}

define fastcc i64 @extract_ri8_v256i64(<256 x i64> %v) {
; CHECK-LABEL: extract_ri8_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x i64> %v, i32 128
  ret i64 %ret
}

define fastcc i64 @extract_ri_v512i64(<512 x i64> %v) {
; CHECK-LABEL: extract_ri_v512i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v1(116)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <512 x i64> %v, i32 372
  ret i64 %ret
}

;;; <256 x i32>

define fastcc i32 @extract_rr_v256i32(i32 signext %idx, <256 x i32> %v) {
; CHECK-LABEL: extract_rr_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x i32> %v, i32 %idx
  ret i32 %ret
}

define fastcc i32 @extract_ri7_v256i32(<256 x i32> %v) {
; CHECK-LABEL: extract_ri7_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(127)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x i32> %v, i32 127
  ret i32 %ret
}

define fastcc i32 @extract_ri8_v256i32(<256 x i32> %v) {
; CHECK-LABEL: extract_ri8_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x i32> %v, i32 128
  ret i32 %ret
}

define fastcc i32 @extract_ri_v512i32(<512 x i32> %v) {
; CHECK-LABEL: extract_ri_v512i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 186
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    srl %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <512 x i32> %v, i32 372
  ret i32 %ret
}

define fastcc i32 @extract_rr_v512i32(<512 x i32> %v, i32 signext %idx) {
; CHECK-LABEL: extract_rr_v512i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srl %s1, %s0, 1
; CHECK-NEXT:    lvs %s1, %v0(%s1)
; CHECK-NEXT:    nnd %s0, %s0, (63)0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    srl %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <512 x i32> %v, i32 %idx
  ret i32 %ret
}

;;; <256 x double>

define fastcc double @extract_rr_v256f64(i32 signext %idx, <256 x double> %v) {
; CHECK-LABEL: extract_rr_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x double> %v, i32 %idx
  ret double %ret
}

define fastcc double @extract_ri7_v256f64(<256 x double> %v) {
; CHECK-LABEL: extract_ri7_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(127)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x double> %v, i32 127
  ret double %ret
}

define fastcc double @extract_ri8_v256f64(<256 x double> %v) {
; CHECK-LABEL: extract_ri8_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x double> %v, i32 128
  ret double %ret
}

define fastcc double @extract_ri_v512f64(<512 x double> %v) {
; CHECK-LABEL: extract_ri_v512f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v1(116)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <512 x double> %v, i32 372
  ret double %ret
}

;;; <256 x float>

define fastcc float @extract_rr_v256f32(i32 signext %idx, <256 x float> %v) {
; CHECK-LABEL: extract_rr_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x float> %v, i32 %idx
  ret float %ret
}

define fastcc float @extract_ri7_v256f32(<256 x float> %v) {
; CHECK-LABEL: extract_ri7_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvs %s0, %v0(127)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x float> %v, i32 127
  ret float %ret
}

define fastcc float @extract_ri8_v256f32(<256 x float> %v) {
; CHECK-LABEL: extract_ri8_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 128
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <256 x float> %v, i32 128
  ret float %ret
}

define fastcc float @extract_ri_v512f32(<512 x float> %v) {
; CHECK-LABEL: extract_ri_v512f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 186
; CHECK-NEXT:    lvs %s0, %v0(%s0)
; CHECK-NEXT:    srl %s0, %s0, 32
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <512 x float> %v, i32 372
  ret float %ret
}

define fastcc float @extract_rr_v512f32(<512 x float> %v, i32 signext %idx) {
; CHECK-LABEL: extract_rr_v512f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srl %s1, %s0, 1
; CHECK-NEXT:    lvs %s1, %v0(%s1)
; CHECK-NEXT:    nnd %s0, %s0, (63)0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    srl %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = extractelement <512 x float> %v, i32 %idx
  ret float %ret
}
