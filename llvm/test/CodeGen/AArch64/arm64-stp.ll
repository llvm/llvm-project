; RUN: llc < %s -mtriple=arm64-eabi -aarch64-enable-stp-suppress=false -verify-machineinstrs -mcpu=cyclone | FileCheck %s

; CHECK-LABEL: stp_int
; CHECK: stp w0, w1, [x2]
define void @stp_int(i32 %a, i32 %b, i32* nocapture %p) nounwind {
  store i32 %a, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  store i32 %b, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: stp_long
; CHECK: stp x0, x1, [x2]
define void @stp_long(i64 %a, i64 %b, i64* nocapture %p) nounwind {
  store i64 %a, i64* %p, align 8
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 1
  store i64 %b, i64* %add.ptr, align 8
  ret void
}

; CHECK-LABEL: stp_float
; CHECK: stp s0, s1, [x0]
define void @stp_float(float %a, float %b, float* nocapture %p) nounwind {
  store float %a, float* %p, align 4
  %add.ptr = getelementptr inbounds float, float* %p, i64 1
  store float %b, float* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: stp_double
; CHECK: stp d0, d1, [x0]
define void @stp_double(double %a, double %b, double* nocapture %p) nounwind {
  store double %a, double* %p, align 8
  %add.ptr = getelementptr inbounds double, double* %p, i64 1
  store double %b, double* %add.ptr, align 8
  ret void
}

; CHECK-LABEL: stp_doublex2
; CHECK: stp q0, q1, [x0]
define void @stp_doublex2(<2 x double> %a, <2 x double> %b, <2 x double>* nocapture %p) nounwind {
  store <2 x double> %a, <2 x double>* %p, align 16
  %add.ptr = getelementptr inbounds <2 x double>, <2 x double>* %p, i64 1
  store <2 x double> %b, <2 x double>* %add.ptr, align 16
  ret void
}

; Test the load/store optimizer---combine ldurs into a ldp, if appropriate
define void @stur_int(i32 %a, i32 %b, i32* nocapture %p) nounwind {
; CHECK-LABEL: stur_int
; CHECK: stp w{{[0-9]+}}, {{w[0-9]+}}, [x{{[0-9]+}}, #-8]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %p, i32 -1
  store i32 %a, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %p, i32 -2
  store i32 %b, i32* %p2, align 2
  ret void
}

define void @stur_long(i64 %a, i64 %b, i64* nocapture %p) nounwind {
; CHECK-LABEL: stur_long
; CHECK: stp x{{[0-9]+}}, {{x[0-9]+}}, [x{{[0-9]+}}, #-16]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %p, i32 -1
  store i64 %a, i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %p, i32 -2
  store i64 %b, i64* %p2, align 2
  ret void
}

define void @stur_float(float %a, float %b, float* nocapture %p) nounwind {
; CHECK-LABEL: stur_float
; CHECK: stp s{{[0-9]+}}, {{s[0-9]+}}, [x{{[0-9]+}}, #-8]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds float, float* %p, i32 -1
  store float %a, float* %p1, align 2
  %p2 = getelementptr inbounds float, float* %p, i32 -2
  store float %b, float* %p2, align 2
  ret void
}

define void @stur_double(double %a, double %b, double* nocapture %p) nounwind {
; CHECK-LABEL: stur_double
; CHECK: stp d{{[0-9]+}}, {{d[0-9]+}}, [x{{[0-9]+}}, #-16]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds double, double* %p, i32 -1
  store double %a, double* %p1, align 2
  %p2 = getelementptr inbounds double, double* %p, i32 -2
  store double %b, double* %p2, align 2
  ret void
}

define void @stur_doublex2(<2 x double> %a, <2 x double> %b, <2 x double>* nocapture %p) nounwind {
; CHECK-LABEL: stur_doublex2
; CHECK: stp q{{[0-9]+}}, q{{[0-9]+}}, [x{{[0-9]+}}, #-32]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 -1
  store <2 x double> %a, <2 x double>* %p1, align 2
  %p2 = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 -2
  store <2 x double> %b, <2 x double>* %p2, align 2
  ret void
}

define void @splat_v4i32(i32 %v, i32 *%p) {
entry:

; CHECK-LABEL: splat_v4i32
; CHECK-DAG: dup v0.4s, w0
; CHECK-DAG: str q0, [x1]
; CHECK: ret

  %p17 = insertelement <4 x i32> undef, i32 %v, i32 0
  %p18 = insertelement <4 x i32> %p17, i32 %v, i32 1
  %p19 = insertelement <4 x i32> %p18, i32 %v, i32 2
  %p20 = insertelement <4 x i32> %p19, i32 %v, i32 3
  %p21 = bitcast i32* %p to <4 x i32>*
  store <4 x i32> %p20, <4 x i32>* %p21, align 4
  ret void
}

; Check that a non-splat store that is storing a vector created by 4
; insertelements that is not a splat vector does not get split.
define void @nosplat_v4i32(i32 %v, i32 *%p) {
entry:

; CHECK-LABEL: nosplat_v4i32:
; CHECK: str w0,
; CHECK: ldr q[[REG1:[0-9]+]],
; CHECK-DAG: mov v[[REG1]].s[1], w0
; CHECK-DAG: mov v[[REG1]].s[2], w0
; CHECK-DAG: mov v[[REG1]].s[3], w0
; CHECK: str q[[REG1]], [x1]
; CHECK: ret

  %p17 = insertelement <4 x i32> undef, i32 %v, i32 %v
  %p18 = insertelement <4 x i32> %p17, i32 %v, i32 1
  %p19 = insertelement <4 x i32> %p18, i32 %v, i32 2
  %p20 = insertelement <4 x i32> %p19, i32 %v, i32 3
  %p21 = bitcast i32* %p to <4 x i32>*
  store <4 x i32> %p20, <4 x i32>* %p21, align 4
  ret void
}

; Check that a non-splat store that is storing a vector created by 4
; insertelements that is not a splat vector does not get split.
define void @nosplat2_v4i32(i32 %v, i32 *%p, <4 x i32> %vin) {
entry:

; CHECK-LABEL: nosplat2_v4i32:
; CHECK: mov v[[REG1]].s[1], w0
; CHECK-DAG: mov v[[REG1]].s[2], w0
; CHECK-DAG: mov v[[REG1]].s[3], w0
; CHECK: str q[[REG1]], [x1]
; CHECK: ret

  %p18 = insertelement <4 x i32> %vin, i32 %v, i32 1
  %p19 = insertelement <4 x i32> %p18, i32 %v, i32 2
  %p20 = insertelement <4 x i32> %p19, i32 %v, i32 3
  %p21 = bitcast i32* %p to <4 x i32>*
  store <4 x i32> %p20, <4 x i32>* %p21, align 4
  ret void
}

; Read of %b to compute %tmp2 shouldn't prevent formation of stp
; CHECK-LABEL: stp_int_rar_hazard
; CHECK: ldr [[REG:w[0-9]+]], [x2, #8]
; CHECK: add w8, [[REG]], w1
; CHECK: stp w0, w1, [x2]
; CHECK: ret
define i32 @stp_int_rar_hazard(i32 %a, i32 %b, i32* nocapture %p) nounwind {
  store i32 %a, i32* %p, align 4
  %ld.ptr = getelementptr inbounds i32, i32* %p, i64 2
  %tmp = load i32, i32* %ld.ptr, align 4
  %tmp2 = add i32 %tmp, %b
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  store i32 %b, i32* %add.ptr, align 4
  ret i32 %tmp2
}

; Read of %b to compute %tmp2 shouldn't prevent formation of stp
; CHECK-LABEL: stp_int_rar_hazard_after
; CHECK: ldr [[REG:w[0-9]+]], [x3, #4]
; CHECK: add w0, [[REG]], w2
; CHECK: stp w1, w2, [x3]
; CHECK: ret
define i32 @stp_int_rar_hazard_after(i32 %w0, i32 %a, i32 %b, i32* nocapture %p) nounwind {
  store i32 %a, i32* %p, align 4
  %ld.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp = load i32, i32* %ld.ptr, align 4
  %tmp2 = add i32 %tmp, %b
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  store i32 %b, i32* %add.ptr, align 4
  ret i32 %tmp2
}
