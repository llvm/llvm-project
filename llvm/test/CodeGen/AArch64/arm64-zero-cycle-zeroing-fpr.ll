; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+no-zcz-fpr64 | FileCheck %s -check-prefixes=ALL,NOZCZ-FPR64-NOZCZ-FPR128
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+no-zcz-fpr64,+fullfp16 | FileCheck %s -check-prefixes=ALL,NOZCZ-FPR64-NOZCZ-FPR128-FULLFP16
; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s -check-prefixes=ALL,ZCZ-FPR64
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+fullfp16 | FileCheck %s -check-prefixes=ALL,ZCZ-FPR64
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+no-zcz-fpr64,+zcz-fpr128 | FileCheck %s -check-prefixes=ALL,NOZCZ-FPR64-ZCZ-FPR128
; RUN: llc < %s -mtriple=arm64-apple-ios -mcpu=cyclone | FileCheck %s -check-prefixes=ALL,FP-WORKAROUND
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=ALL,NOZCZ-FPR64-ZCZ-FPR128
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=exynos-m3 | FileCheck %s -check-prefixes=ALL,ZCZ-FPR64
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=kryo | FileCheck %s -check-prefixes=ALL,ZCZ-FPR64
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=falkor | FileCheck %s -check-prefixes=ALL,ZCZ-FPR64

define half @tf16() {
entry:
; ALL-LABEL: tf16:
; FP-WORKAROUND: mov s0, wzr
; NOZCZ-FPR64-NOZCZ-FPR128: mov s0, wzr
; NOZCZ-FPR64-NOZCZ-FPR128-FULLFP16: mov h0, wzr
; ZCZ-FPR64: movi d0, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret half 0.0
}

define float @tf32() {
entry:
; ALL-LABEL: tf32:
; FP-WORKAROUND: mov s0, wzr
; NOZCZ-FPR64-NOZCZ-FPR128: mov s0, wzr
; ZCZ-FPR64: movi d0, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret float 0.0
}

define double @td64() {
entry:
; ALL-LABEL: td64:
; FP-WORKAROUND: mov d0, xzr
; NOZCZ-FPR64-NOZCZ-FPR128: mov d0, xzr
; ZCZ-FPR64: movi d0, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret double 0.0
}

define <8 x i8> @tv8i8() {
entry:
; ALL-LABEL: tv8i8:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
}

define <4 x i16> @tv4i16() {
entry:
; ALL-LABEL: tv4i16:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <4 x i16> <i16 0, i16 0, i16 0, i16 0>
}

define <2 x i32> @tv2i32() {
entry:
; ALL-LABEL: tv2i32:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <2 x i32> <i32 0, i32 0>
}

define <2 x float> @tv2f32() {
entry:
; ALL-LABEL: tv2f32:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <2 x float> <float 0.0, float 0.0>
}

define <16 x i8> @tv16i8() {
entry:
; ALL-LABEL: tv16i8:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
}

define <8 x i16> @tv8i16() {
entry:
; ALL-LABEL: tv8i16:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
}

define <4 x i32> @tv4i32() {
entry:
; ALL-LABEL: tv4i32:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <4 x i32> <i32 0, i32 0, i32 0, i32 0>
}

define <2 x i64> @tv2i64() {
entry:
; ALL-LABEL: tv2i64:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <2 x i64> <i64 0, i64 0>
}

define <4 x float> @tv4f32() {
entry:
; ALL-LABEL: tv4f32:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>
}

define <2 x double> @tv2d64() {
entry:
; ALL-LABEL: tv2d64:
; FP-WORKAROUND: movi{{(.16b)?}} v0{{(.16b)?}}, #0
; NOZCZ-FPR64-NOZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; ZCZ-FPR64: movi{{(.2d)?}} v0{{(.2d)?}}, #0
; NOZCZ-FPR64-ZCZ-FPR128: movi{{(.2d)?}} v0{{(.2d)?}}, #0
  ret <2 x double> <double 0.0, double 0.0>
}

; We used to produce spills+reloads for a Q register with zero cycle zeroing
; enabled.
; ALL-LABEL: foo:
; ALL-NOT: str q{{[0-9]+}}
; ALL-NOT: ldr q{{[0-9]+}}
define double @foo(i32 %n) {
entry:
  br label %for.body

for.body:
  %phi0 = phi double [ 1.0, %entry ], [ %v0, %for.body ]
  %i.076 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %conv21 = sitofp i32 %i.076 to double
  %call = tail call fast double @sin(double %conv21)
  %cmp.i = fcmp fast olt double %phi0, %call
  %v0 = select i1 %cmp.i, double %call, double %phi0
  %inc = add nuw nsw i32 %i.076, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret double %v0
}

declare double @sin(double)
