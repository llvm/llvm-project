; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=soft-float -O3 < %s | FileCheck %s
;
; Test that arguments and return values of fp/vector types are always handled
; with gprs with soft-float.

define double @f1(double %arg) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK-NOT: %{{[fv]}}
; CHECK: llihh %r3, 16368
; CHECK-NEXT: brasl %r14, __adddf3@PLT
; CHECK-NEXT: lmg   %r14, %r15, 272(%r15)
; CHECK-NEXT: br    %r14
  %res = fadd double %arg, 1.0
  ret double %res
}

define float @f2(float %arg) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK-NOT: %{{[fv]}}
; CHECK: llgfr   %r2, %r2
; CHECK-NEXT: llilh   %r3, 16256
; CHECK-NEXT: brasl   %r14, __addsf3@PLT
; CHECK-NEXT: # kill: def $r2l killed $r2l killed $r2d
; CHECK-NEXT: lmg     %r14, %r15, 272(%r15)
; CHECK-NEXT: br      %r14
  %res = fadd float %arg, 1.0
  ret float %res
}

define fp128 @f2_fp128(fp128 %arg) {
; CHECK-LABEL: f2_fp128:
; CHECK-NOT: %{{[fv]}}
; CHECK: aghi    %r15, -208
; CHECK-NEXT: .cfi_def_cfa_offset 368
; CHECK-NEXT: lg      %r0, 0(%r2)
; CHECK-NEXT: lg      %r1, 8(%r2)
; CHECK-NEXT: llihf   %r2, 1073823744
; CHECK-NEXT: stg     %r2, 160(%r15)
; CHECK-NEXT: la      %r2, 192(%r15)
; CHECK-NEXT: la      %r3, 176(%r15)
; CHECK-NEXT: la      %r4, 160(%r15)
; CHECK-NEXT: stg     %r1, 184(%r15)
; CHECK-NEXT: stg     %r0, 176(%r15)
; CHECK-NEXT: mvghi   168(%r15), 0
; CHECK-NEXT: brasl   %r14, __addtf3@PLT
; CHECK-NEXT: lg      %r2, 192(%r15)
; CHECK-NEXT: lg      %r3, 200(%r15)
; CHECK-NEXT: lmg     %r14, %r15, 320(%r15)
; CHECK-NEXT: br      %r14
  %res = fadd fp128 %arg, 0xL00000000000000004001400000000000
  ret fp128 %res
}

define <2 x double> @f3(<2 x double> %arg) {
; CHECK-LABEL: f3:
; CHECK-NOT: %{{[fv]}}
; CHECK: lg      %r13, 8(%r2)
; CHECK-NEXT: lg      %r2, 0(%r2)
; CHECK-NEXT: llihh   %r3, 16368
; CHECK-NEXT: brasl   %r14, __adddf3@PLT
; CHECK-NEXT: lgr     %r12, %r2
; CHECK-NEXT: lgr     %r2, %r13
; CHECK-NEXT: llihh   %r3, 16368
; CHECK-NEXT: brasl   %r14, __adddf3@PLT
; CHECK-NEXT: lgr     %r3, %r2
; CHECK-NEXT: lgr     %r2, %r12
; CHECK-NEXT: lmg     %r12, %r15, 256(%r15)
; CHECK-NEXT: br      %r14
  %res = fadd <2 x double> %arg, <double 1.000000e+00, double 1.000000e+00>
  ret <2 x double> %res
}

define <2 x float> @f4(<2 x float> %arg) {
; CHECK-LABEL: f4:
; CHECK-NOT: %{{[fv]}}
; CHECK: lr      %r13, %r3
; CHECK-NEXT: llgfr   %r2, %r2
; CHECK-NEXT: llilh   %r3, 16256
; CHECK-NEXT: brasl   %r14, __addsf3@PLT
; CHECK-NEXT: lgr     %r12, %r2
; CHECK-NEXT: llgfr   %r2, %r13
; CHECK-NEXT: llilh   %r3, 16256
; CHECK-NEXT: brasl   %r14, __addsf3@PLT
; CHECK-NEXT: lgr     %r3, %r2
; CHECK-NEXT: lr      %r2, %r12
; CHECK-NEXT: # kill: def $r3l killed $r3l killed $r3d
; CHECK-NEXT: lmg     %r12, %r15, 256(%r15)
; CHECK-NEXT: br      %r14
  %res = fadd <2 x float> %arg, <float 1.000000e+00, float 1.000000e+00>
  ret <2 x float> %res
}

define <2 x i64> @f5(<2 x i64> %arg) {
; CHECK-LABEL: f5:
; CHECK-NOT: %{{[fv]}}
; CHECK: lghi    %r0, 1
; CHECK-NEXT: ag      %r0, 0(%r2)
; CHECK-NEXT: lghi    %r3, 1
; CHECK-NEXT: ag      %r3, 8(%r2)
; CHECK-NEXT: lgr     %r2, %r0
; CHECK-NEXT: br      %r14
  %res = add <2 x i64> %arg, <i64 1, i64 1>
  ret <2 x i64> %res
}

define <2 x i32> @f6(<2 x i32> %arg) {
; CHECK-LABEL: f6:
; CHECK-NOT: %{{[fv]}}
; CHECK: ahi     %r2, 1
; CHECK-NEXT: ahi     %r3, 1
; CHECK-NEXT: br      %r14
  %res = add <2 x i32> %arg, <i32 1, i32 1>
  ret <2 x i32> %res
}

;; Stack arguments

define double @f7(double %A, double %B, double %C, double %D, double %E,
                  double %F) {
; CHECK-LABEL: f7:
; CHECK-NOT: %{{[fv]}}
; CHECK: aghi    %r15, -160
; CHECK-NEXT: .cfi_def_cfa_offset 320
; CHECK-NEXT: lg      %r3, 320(%r15)
; CHECK-NEXT: brasl   %r14, __adddf3@PLT
; CHECK-NEXT: lmg     %r14, %r15, 272(%r15)
; CHECK-NEXT: br      %r14

  %res = fadd double %A, %F
  ret double %res
}

define float @f8(float %A, float %B, float %C, float %D, float %E,
                 float %F) {
; CHECK-LABEL: f8:
; CHECK-NOT: %{{[fv]}}
; CHECK: aghi    %r15, -160
; CHECK-NEXT: .cfi_def_cfa_offset 320
; CHECK-NEXT: llgf    %r3, 324(%r15)
; CHECK-NEXT: llgfr   %r2, %r2
; CHECK-NEXT: brasl   %r14, __addsf3@PLT
; CHECK-NEXT: # kill: def $r2l killed $r2l killed $r2d
; CHECK-NEXT: lmg     %r14, %r15, 272(%r15)
; CHECK-NEXT: br      %r14
  %res = fadd float %A, %F
  ret float %res
}

define <2 x double> @f9(<2 x double> %A, <2 x double> %B, <2 x double> %C,
                        <2 x double> %D, <2 x double> %E, <2 x double> %F,
                        <2 x double> %G, <2 x double> %H, <2 x double> %I) {
; CHECK-LABEL: f9:
; CHECK-NOT: %{{[fv]}}
; CHECK: aghi    %r15, -160
; CHECK-NEXT: .cfi_def_cfa_offset 320
; CHECK-NEXT: lg      %r1, 344(%r15)
; CHECK-NEXT: lg      %r13, 8(%r2)
; CHECK-NEXT: lg      %r2, 0(%r2)
; CHECK-NEXT: lg      %r3, 0(%r1)
; CHECK-NEXT: lg      %r12, 8(%r1)
; CHECK-NEXT: brasl   %r14, __adddf3@PLT
; CHECK-NEXT: lgr     %r11, %r2
; CHECK-NEXT: lgr     %r2, %r13
; CHECK-NEXT: lgr     %r3, %r12
; CHECK-NEXT: brasl   %r14, __adddf3@PLT
; CHECK-NEXT: lgr     %r3, %r2
; CHECK-NEXT: lgr     %r2, %r11
; CHECK-NEXT: lmg     %r11, %r15, 248(%r15)
; CHECK-NEXT: br      %r14
  %res = fadd <2 x double> %A, %I
  ret <2 x double> %res
}

define <2 x float> @f10(<2 x float> %A, <2 x float> %B, <2 x float> %C,
                        <2 x float> %D, <2 x float> %E, <2 x float> %F,
                        <2 x float> %G, <2 x float> %H, <2 x float> %I) {
; CHECK-LABEL: f10:
; CHECK-NOT: %{{[fv]}}
; CHECK: aghi    %r15, -160
; CHECK-NEXT: .cfi_def_cfa_offset 320
; CHECK-NEXT: lr      %r13, %r3
; CHECK-NEXT: llgf    %r3, 412(%r15)
; CHECK-NEXT: llgf    %r12, 420(%r15)
; CHECK-NEXT: llgfr   %r2, %r2
; CHECK-NEXT: brasl   %r14, __addsf3@PLT
; CHECK-NEXT: lgr     %r11, %r2
; CHECK-NEXT: llgfr   %r2, %r13
; CHECK-NEXT: lgr     %r3, %r12
; CHECK-NEXT: brasl   %r14, __addsf3@PLT
; CHECK-NEXT: lgr     %r3, %r2
; CHECK-NEXT: lr      %r2, %r11
; CHECK-NEXT: # kill: def $r3l killed $r3l killed $r3d
; CHECK-NEXT: lmg     %r11, %r15, 248(%r15)
; CHECK-NEXT: br      %r14

  %res = fadd <2 x float> %A, %I
  ret <2 x float> %res
}

define <2 x i64> @f11(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C,
                      <2 x i64> %D, <2 x i64> %E, <2 x i64> %F,
                      <2 x i64> %G, <2 x i64> %H, <2 x i64> %I) {
; CHECK-LABEL: f11:
; CHECK-NOT: %{{[fv]}}
; CHECK: lg      %r1, 184(%r15)
; CHECK-NEXT: lg      %r3, 8(%r2)
; CHECK-NEXT: lg      %r2, 0(%r2)
; CHECK-NEXT: ag      %r2, 0(%r1)
; CHECK-NEXT: ag      %r3, 8(%r1)
; CHECK-NEXT: br      %r14
  %res = add <2 x i64> %A, %I
  ret <2 x i64> %res
}

;; calls

declare double @bar_double(double %arg);
define double @f12(double %arg, double %arg2) {
; CHECK-LABEL: f12:
; CHECK-NOT: %{{[fv]}}
; CHECK-NOT: %r{{[23]}}
; CHECK: lgr     %r2, %r3
; CHECK-NEXT: brasl   %r14, bar_double@PLT
; CHECK-NEXT: lmg     %r14, %r15, 272(%r15)
; CHECK-NEXT: br      %r14
  %res = call double @bar_double(double %arg2)
  ret double %res
}

declare float @bar_float(float %arg);
define float @f13(float %arg, float %arg2) {
; CHECK-LABEL: f13:
; CHECK-NOT: %{{[fv]}}
; CHECK-NOT: %r{{[23]}}
; CHECK: lr     %r2, %r3
; CHECK-NEXT: brasl   %r14, bar_float@PLT
; CHECK-NEXT: lmg     %r14, %r15, 272(%r15)
; CHECK-NEXT: br      %r14
  %res = call float @bar_float(float %arg2)
  ret float %res
}

declare fp128 @bar_fp128(fp128 %arg);
define fp128 @f14(fp128 %arg, fp128 %arg2) {
; CHECK-LABEL: f14:
; CHECK-NOT: %{{[fv]}}
; CHECK-NOT: %r3
; CHECK: lg      %r0, 0(%r3)
; CHECK-NEXT: lg      %r1, 8(%r3)
; CHECK-NEXT: la      %r2, 160(%r15)
; CHECK-NEXT: stg     %r1, 168(%r15)
; CHECK-NEXT: stg     %r0, 160(%r15)
; CHECK-NEXT: brasl   %r14, bar_fp128@PLT
; CHECK-NEXT: lmg     %r14, %r15, 288(%r15)
; CHECK-NEXT: br      %r14
  %res = call fp128 @bar_fp128(fp128 %arg2)
  ret fp128 %res
}

declare <2 x double> @bar_v2f64(<2 x double> %arg);
define <2 x double> @f15(<2 x double> %arg, <2 x double> %arg2) {
; CHECK-LABEL: f15:
; CHECK-NOT: %{{[fv]}}
; CHECK-NOT: %r3
; CHECK: lg      %r0, 0(%r3)
; CHECK-NEXT: lg      %r1, 8(%r3)
; CHECK-NEXT: la      %r2, 160(%r15)
; CHECK-NEXT: stg     %r1, 168(%r15)
; CHECK-NEXT: stg     %r0, 160(%r15)
; CHECK-NEXT: brasl   %r14, bar_v2f64@PLT
; CHECK-NEXT: lmg     %r14, %r15, 288(%r15)
; CHECK-NEXT: br      %r14
  %res = call <2 x double> @bar_v2f64(<2 x double> %arg2)
  ret <2 x double> %res
}

declare <2 x float> @bar_v2f32(<2 x float> %arg);
define <2 x float> @f16(<2 x float> %arg, <2 x float> %arg2) {
; CHECK-LABEL: f16:
; CHECK-NOT: %{{[fv]}}
; CHECK-NOT: %r{{[2345]}}
; CHECK: lr      %r3, %r5
; CHECK-NEXT: lr      %r2, %r4
; CHECK-NEXT: brasl   %r14, bar_v2f32@PLT
; CHECK-NEXT: lmg     %r14, %r15, 272(%r15)
; CHECK-NEXT: br      %r14
  %res = call <2 x float> @bar_v2f32(<2 x float> %arg2)
  ret <2 x float> %res
}

declare <2 x i64> @bar_v2i64(<2 x i64> %arg);
define <2 x i64> @f17(<2 x i64> %arg, <2 x i64> %arg2) {
; CHECK-LABEL: f17:
; CHECK-NOT: %{{[fv]}}
; CHECK-NOT: %r3
; CHECK: lg      %r0, 0(%r3)
; CHECK-NEXT: lg      %r1, 8(%r3)
; CHECK-NEXT: la      %r2, 160(%r15)
; CHECK-NEXT: stg     %r1, 168(%r15)
; CHECK-NEXT: stg     %r0, 160(%r15)
; CHECK-NEXT: brasl   %r14, bar_v2i64@PLT
; CHECK-NEXT: lmg     %r14, %r15, 288(%r15)
; CHECK-NEXT: br      %r14
  %res = call <2 x i64> @bar_v2i64(<2 x i64> %arg2)
  ret <2 x i64> %res
}
