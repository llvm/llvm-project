; RUN: llc < %s -mtriple=ve-unknown-unknown -mattr=+vpu | FileCheck %s


;;; <256 x i64>

define fastcc <256 x i64> @insert_rr_v256i64(i32 signext %idx, i64 %s) {
; CHECK-LABEL: insert_rr_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v0(%s0), %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x i64> undef, i64 %s, i32 %idx
  ret <256 x i64> %ret
}

define fastcc <256 x i64> @insert_ri7_v256i64(i64 %s) {
; CHECK-LABEL: insert_ri7_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v0(127), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x i64> undef, i64 %s, i32 127
  ret <256 x i64> %ret
}

define fastcc <256 x i64> @insert_ri8_v256i64(i64 %s) {
; CHECK-LABEL: insert_ri8_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lsv %v0(%s1), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x i64> undef, i64 %s, i32 128
  ret <256 x i64> %ret
}

define fastcc <512 x i64> @insert_ri_v512i64(i64 %s) {
; CHECK-LABEL: insert_ri_v512i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v1(116), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <512 x i64> undef, i64 %s, i32 372
  ret <512 x i64> %ret
}

;;; <256 x i32>

define fastcc <256 x i32> @insert_rr_v256i32(i32 signext %idx, i32 signext %s) {
; CHECK-LABEL: insert_rr_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lsv %v0(%s0), %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x i32> undef, i32 %s, i32 %idx
  ret <256 x i32> %ret
}

define fastcc <256 x i32> @insert_ri7_v256i32(i32 signext %s) {
; CHECK-LABEL: insert_ri7_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lsv %v0(127), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x i32> undef, i32 %s, i32 127
  ret <256 x i32> %ret
}

define fastcc <256 x i32> @insert_ri8_v256i32(i32 signext %s) {
; CHECK-LABEL: insert_ri8_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lsv %v0(%s1), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x i32> undef, i32 %s, i32 128
  ret <256 x i32> %ret
}

define fastcc <512 x i32> @insert_ri_v512i32(i32 signext %s) {
; CHECK-LABEL: insert_ri_v512i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 186
; CHECK-NEXT:    lvs %s2, %v0(%s1)
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    or %s0, %s2, %s0
; CHECK-NEXT:    lsv %v0(%s1), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <512 x i32> undef, i32 %s, i32 372
  ret <512 x i32> %ret
}

define fastcc <512 x i32> @insert_rr_v512i32(i32 signext %idx, i32 signext %s) {
; CHECK-LABEL: insert_rr_v512i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    nnd %s2, %s0, (63)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, 5
; CHECK-NEXT:    sll %s1, %s1, %s2
; CHECK-NEXT:    srl %s0, %s0, 1
; CHECK-NEXT:    lvs %s3, %v0(%s0)
; CHECK-NEXT:    srl %s2, (32)1, %s2
; CHECK-NEXT:    and %s2, %s3, %s2
; CHECK-NEXT:    or %s1, %s2, %s1
; CHECK-NEXT:    lsv %v0(%s0), %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <512 x i32> undef, i32 %s, i32 %idx
  ret <512 x i32> %ret
}

;;; <256 x double>

define fastcc <256 x double> @insert_rr_v256f64(i32 signext %idx, double %s) {
; CHECK-LABEL: insert_rr_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v0(%s0), %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x double> undef, double %s, i32 %idx
  ret <256 x double> %ret
}

define fastcc <256 x double> @insert_ri7_v256f64(double %s) {
; CHECK-LABEL: insert_ri7_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v0(127), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x double> undef, double %s, i32 127
  ret <256 x double> %ret
}

define fastcc <256 x double> @insert_ri8_v256f64(double %s) {
; CHECK-LABEL: insert_ri8_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lsv %v0(%s1), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x double> undef, double %s, i32 128
  ret <256 x double> %ret
}

define fastcc <512 x double> @insert_ri_v512f64(double %s) {
; CHECK-LABEL: insert_ri_v512f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v1(116), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <512 x double> undef, double %s, i32 372
  ret <512 x double> %ret
}

;;; <256 x float>

define fastcc <256 x float> @insert_rr_v256f32(i32 signext %idx, float %s) {
; CHECK-LABEL: insert_rr_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v0(%s0), %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x float> undef, float %s, i32 %idx
  ret <256 x float> %ret
}

define fastcc <256 x float> @insert_ri7_v256f32(float %s) {
; CHECK-LABEL: insert_ri7_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lsv %v0(127), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x float> undef, float %s, i32 127
  ret <256 x float> %ret
}

define fastcc <256 x float> @insert_ri8_v256f32(float %s) {
; CHECK-LABEL: insert_ri8_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    lsv %v0(%s1), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <256 x float> undef, float %s, i32 128
  ret <256 x float> %ret
}

define fastcc <512 x float> @insert_ri_v512f32(float %s) {
; CHECK-LABEL: insert_ri_v512f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    lea %s1, 186
; CHECK-NEXT:    lvs %s2, %v0(%s1)
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    or %s0, %s2, %s0
; CHECK-NEXT:    lsv %v0(%s1), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <512 x float> undef, float %s, i32 372
  ret <512 x float> %ret
}

define fastcc <512 x float> @insert_rr_v512f32(i32 signext %idx, float %s) {
; CHECK-LABEL: insert_rr_v512f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s1, 32
; CHECK-NEXT:    srl %s2, %s0, 1
; CHECK-NEXT:    lvs %s3, %v0(%s2)
; CHECK-NEXT:    nnd %s0, %s0, (63)0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    srl %s4, (32)1, %s0
; CHECK-NEXT:    and %s3, %s3, %s4
; CHECK-NEXT:    adds.w.zx %s1, %s1, (0)1
; CHECK-NEXT:    sll %s0, %s1, %s0
; CHECK-NEXT:    or %s0, %s3, %s0
; CHECK-NEXT:    lsv %v0(%s2), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = insertelement <512 x float> undef, float %s, i32 %idx
  ret <512 x float> %ret
}
