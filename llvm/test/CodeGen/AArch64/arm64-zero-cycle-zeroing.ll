; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=-zcz-gp,+no-zcz-fp      | FileCheck %s -check-prefixes=ALL,NONEGP,NONEFP
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+zcz                    | FileCheck %s -check-prefixes=ALL,ZEROGP,ZEROFP
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+zcz -mattr=+fullfp16   | FileCheck %s -check-prefixes=ALL,ZEROGP,ZERO16
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+zcz-gp,+no-zcz-fp      | FileCheck %s -check-prefixes=ALL,ZEROGP,NONEFP
; RUN: llc < %s -mtriple=aarch64-linux-gnu                                | FileCheck %s -check-prefixes=ALL,NONEGP,ZEROFP
; RUN: llc < %s -mtriple=arm64-apple-ios   -mcpu=cyclone                  | FileCheck %s -check-prefixes=ALL,ZEROGP,NONEFP
; RUN: llc < %s -mtriple=arm64-linux-gnu   -mcpu=apple-a10                | FileCheck %s -check-prefixes=ALL,ZEROGP,ZEROFP
; RUN: llc < %s -mtriple=arm64-apple-ios   -mcpu=cyclone -mattr=+fullfp16 | FileCheck %s -check-prefixes=ALL,ZEROGP,NONE16
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=exynos-m3                | FileCheck %s -check-prefixes=ALL,NONEGP,ZEROFP
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=kryo                     | FileCheck %s -check-prefixes=ALL,ZEROGP,ZEROFP
; UN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=falkor                   | FileCheck %s -check-prefixes=ALL,ZEROGP,ZEROFP

declare void @bar(half, float, double, <2 x double>)
declare void @bari(i32, i32)
declare void @barl(i64, i64)
declare void @barf(float, float)

define void @t1() nounwind ssp {
entry:
; ALL-LABEL: t1:
; ALL-NOT: fmov
; NONEFP: ldr h0,{{.*}}
; NONEFP: fmov s1, wzr
; NONEFP: fmov d2, xzr
; NONEFP: movi{{(.16b)?}} v3{{(.2d)?}}, #0
; NONE16: fmov h0, wzr
; NONE16: fmov s1, wzr
; NONE16: fmov d2, xzr
; NONE16: movi{{(.16b)?}} v3{{(.2d)?}}, #0
; ZEROFP-DAG: ldr h0,{{.*}}
; ZEROFP-DAG: movi d1, #0
; ZEROFP-DAG: movi d2, #0
; ZEROFP-DAG: movi v3.2d, #0
; ZERO16: movi d0, #0
; ZERO16: movi d1, #0
; ZERO16: movi d2, #0
; ZERO16: movi v3.2d, #0
  tail call void @bar(half 0.000000e+00, float 0.000000e+00, double 0.000000e+00, <2 x double> <double 0.000000e+00, double 0.000000e+00>) nounwind
  ret void
}

define void @t2() nounwind ssp {
entry:
; ALL-LABEL: t2:
; NONEGP: mov w0, wzr
; NONEGP: mov w1, wzr
; ZEROGP: mov w0, #0
; ZEROGP: mov w1, #0
  tail call void @bari(i32 0, i32 0) nounwind
  ret void
}

define void @t3() nounwind ssp {
entry:
; ALL-LABEL: t3:
; NONEGP: mov x0, xzr
; NONEGP: mov x1, xzr
; ZEROGP: mov x0, #0
; ZEROGP: mov x1, #0
  tail call void @barl(i64 0, i64 0) nounwind
  ret void
}

define void @t4() nounwind ssp {
; ALL-LABEL: t4:
; NONEFP: fmov s{{[0-3]+}}, wzr
; NONEFP: fmov s{{[0-3]+}}, wzr
; ZEROFP: movi d0, #0
; ZEROFP: movi d1, #0
  tail call void @barf(float 0.000000e+00, float 0.000000e+00) nounwind
  ret void
}

declare double @sin(double)

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

define <2 x i64> @t6() {
; ALL-LABEL: t6:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <2 x i64> zeroinitializer
}

define i1 @ti1() {
entry:
; ALL-LABEL: ti1:
; NONEGP: mov w0, wzr
; ZEROGP: mov w0, #0
  ret i1 false
}

define i8 @ti8() {
entry:
; ALL-LABEL: ti8:
; NONEGP: mov w0, wzr
; ZEROGP: mov w0, #0
  ret i8 0
}

define i16 @ti16() {
entry:
; ALL-LABEL: ti16:
; NONEGP: mov w0, wzr
 ; ZEROGP: mov w0, #0
  ret i16 0
}

define i32 @ti32() {
entry:
; ALL-LABEL: ti32:
; NONEGP: mov w0, wzr
; ZEROGP: mov w0, #0
  ret i32 0
}

define i64 @ti64() {
entry:
; ALL-LABEL: ti64:
; NONEGP: mov x0, xzr
; ZEROGP: mov x0, #0
  ret i64 0
}

define float @tf32() {
entry:
; ALL-LABEL: tf32:
; NONEFP: mov s0, wzr
; ZEROFP: movi d0, #0
  ret float 0.0
}

define double @td64() {
entry:
; ALL-LABEL: td64:
; NONEFP: mov d0, xzr
; ZEROFP: movi d0, #0
  ret double 0.0
}

define <8 x i8> @tv8i8() {
entry:
; ALL-LABEL: tv8i8:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
}

define <4 x i16> @tv4i16() {
entry:
; ALL-LABEL: tv4i16:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <4 x i16> <i16 0, i16 0, i16 0, i16 0>
}

define <2 x i32> @tv2i32() {
entry:
; ALL-LABEL: tv2i32:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <2 x i32> <i32 0, i32 0>
}

define <2 x float> @tv2f32() {
entry:
; ALL-LABEL: tv2f32:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <2 x float> <float 0.0, float 0.0>
}

define <16 x i8> @tv16i8() {
entry:
; ALL-LABEL: tv16i8:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
}

define <8 x i16> @tv8i16() {
entry:
; ALL-LABEL: tv8i16:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
}

define <4 x i32> @tv4i32() {
entry:
; ALL-LABEL: tv4i32:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <4 x i32> <i32 0, i32 0, i32 0, i32 0>
}

define <2 x i64> @tv2i64() {
entry:
; ALL-LABEL: tv2i64:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <2 x i64> <i64 0, i64 0>
}

define <4 x float> @tv4f32() {
entry:
; ALL-LABEL: tv4f32:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>
}

define <2 x double> @tv2d64() {
entry:
; ALL-LABEL: tv2d64:
; ALL: movi{{(.16b)?}} v0{{(.2d)?}}, #0
  ret <2 x double> <double 0.0, double 0.0>
}

