; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

;
; Floating-point scalar convert to signed integer (to nearest with ties to away)
;
define i32 @fcvtas_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtas_1w1s:
;CHECK: fcvtas w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtas.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtas_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtas_1x1s:
;CHECK: fcvtas x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtas.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtas_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtas_1w1d:
;CHECK: fcvtas w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtas.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtas_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtas_1x1d:
;CHECK: fcvtas x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtas.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtas.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtas.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtas.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtas.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to unsigned integer
;
define i32 @fcvtau_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtau_1w1s:
;CHECK: fcvtau w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtau.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtau_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtau_1x1s:
;CHECK: fcvtau x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtau.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtau_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtau_1w1d:
;CHECK: fcvtau w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtau.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtau_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtau_1x1d:
;CHECK: fcvtau x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtau.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtau.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtau.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtau.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtau.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to signed integer (toward -Inf)
;
define i32 @fcvtms_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtms_1w1s:
;CHECK: fcvtms w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtms.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtms_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtms_1x1s:
;CHECK: fcvtms x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtms.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtms_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtms_1w1d:
;CHECK: fcvtms w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtms.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtms_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtms_1x1d:
;CHECK: fcvtms x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtms.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtms.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtms.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtms.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtms.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to unsigned integer (toward -Inf)
;
define i32 @fcvtmu_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtmu_1w1s:
;CHECK: fcvtmu w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtmu.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtmu_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtmu_1x1s:
;CHECK: fcvtmu x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtmu.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtmu_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtmu_1w1d:
;CHECK: fcvtmu w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtmu.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtmu_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtmu_1x1d:
;CHECK: fcvtmu x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtmu.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtmu.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtmu.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtmu.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtmu.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to signed integer (to nearest with ties to even)
;
define i32 @fcvtns_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtns_1w1s:
;CHECK: fcvtns w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtns.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtns_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtns_1x1s:
;CHECK: fcvtns x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtns.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtns_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtns_1w1d:
;CHECK: fcvtns w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtns.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtns_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtns_1x1d:
;CHECK: fcvtns x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtns.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtns.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtns.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtns.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtns.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to unsigned integer (to nearest with ties to even)
;
define i32 @fcvtnu_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtnu_1w1s:
;CHECK: fcvtnu w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtnu.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtnu_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtnu_1x1s:
;CHECK: fcvtnu x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtnu.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtnu_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtnu_1w1d:
;CHECK: fcvtnu w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtnu.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtnu_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtnu_1x1d:
;CHECK: fcvtnu x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtnu.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtnu.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtnu.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtnu.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtnu.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to signed integer (toward +Inf)
;
define i32 @fcvtps_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtps_1w1s:
;CHECK: fcvtps w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtps.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtps_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtps_1x1s:
;CHECK: fcvtps x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtps.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtps_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtps_1w1d:
;CHECK: fcvtps w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtps.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtps_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtps_1x1d:
;CHECK: fcvtps x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtps.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtps.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtps.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtps.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtps.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to unsigned integer (toward +Inf)
;
define i32 @fcvtpu_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtpu_1w1s:
;CHECK: fcvtpu w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtpu.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtpu_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtpu_1x1s:
;CHECK: fcvtpu x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtpu.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtpu_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtpu_1w1d:
;CHECK: fcvtpu w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtpu.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtpu_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtpu_1x1d:
;CHECK: fcvtpu x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtpu.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtpu.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtpu.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtpu.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtpu.i64.f64(double) nounwind readnone

;
;  Floating-point scalar convert to signed integer (toward zero)
;
define i32 @fcvtzs_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtzs_1w1s:
;CHECK: fcvtzs w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtzs.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtzs_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtzs_1x1s:
;CHECK: fcvtzs x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtzs.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtzs_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtzs_1w1d:
;CHECK: fcvtzs w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtzs.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtzs_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtzs_1x1d:
;CHECK: fcvtzs x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtzs.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtzs.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtzs.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtzs.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtzs.i64.f64(double) nounwind readnone

;
; Floating-point scalar convert to unsigned integer (toward zero)
;
define i32 @fcvtzu_1w1s(float %A) nounwind {
;CHECK-LABEL: fcvtzu_1w1s:
;CHECK: fcvtzu w0, s0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtzu.i32.f32(float %A)
	ret i32 %tmp3
}

define i64 @fcvtzu_1x1s(float %A) nounwind {
;CHECK-LABEL: fcvtzu_1x1s:
;CHECK: fcvtzu x0, s0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtzu.i64.f32(float %A)
	ret i64 %tmp3
}

define i32 @fcvtzu_1w1d(double %A) nounwind {
;CHECK-LABEL: fcvtzu_1w1d:
;CHECK: fcvtzu w0, d0
;CHECK-NEXT: ret
	%tmp3 = call i32 @llvm.aarch64.neon.fcvtzu.i32.f64(double %A)
	ret i32 %tmp3
}

define i64 @fcvtzu_1x1d(double %A) nounwind {
;CHECK-LABEL: fcvtzu_1x1d:
;CHECK: fcvtzu x0, d0
;CHECK-NEXT: ret
	%tmp3 = call i64 @llvm.aarch64.neon.fcvtzu.i64.f64(double %A)
	ret i64 %tmp3
}

declare i32 @llvm.aarch64.neon.fcvtzu.i32.f32(float) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtzu.i64.f32(float) nounwind readnone
declare i32 @llvm.aarch64.neon.fcvtzu.i32.f64(double) nounwind readnone
declare i64 @llvm.aarch64.neon.fcvtzu.i64.f64(double) nounwind readnone
