; RUN: opt -mtriple=amdgpu6.00 -passes="require<libcall-lowering-info>,expand-ir-insts<O0>" -verify-each %s -S | FileCheck %s
; RUN: opt -mtriple=amdgpu6.00 -passes="require<libcall-lowering-info>,expand-ir-insts<O2>" -verify-each %s -S | FileCheck %s

; A poison result may be unused, but a branch on poison is immediate UB.
; The constant denominator needs no freeze.
define float @poison_numerator(float %x) {
; CHECK-LABEL: define float @poison_numerator(
; CHECK-NEXT:    [[N:%.*]] = fadd ninf float %x, +inf
; CHECK-NEXT:    [[X:%.*]] = freeze float [[N]]
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float [[X]])
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float 2.000000e+00)
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
; CHECK:         call float @llvm.fabs.f32(float [[X]])
; CHECK:       frem.else:
; CHECK-NEXT:    [[ZERO:%.*]] = call float @llvm.copysign.f32(float 0.000000e+00, float [[X]])
; CHECK-NEXT:    [[EQ:%.*]] = fcmp oeq float [[AX]], [[AY]]
; CHECK-NEXT:    select i1 [[EQ]], float [[ZERO]], float [[X]]
; CHECK:       frem.loop_exit:
; CHECK:         call float @llvm.copysign.f32(float {{%.*}}, float [[X]])
  %n = fadd ninf float %x, 0x7FF0000000000000
  %r = frem float %n, 2.0
  ret float %r
}

define float @poison_denominator(float %y) {
; CHECK-LABEL: define float @poison_denominator(
; CHECK-NEXT:    [[N:%.*]] = fadd ninf float %y, +inf
; CHECK-NEXT:    [[Y:%.*]] = freeze float [[N]]
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float 0.000000e+00)
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float [[Y]])
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
; CHECK:         fcmp ueq float [[Y]], 0.000000e+00
  %n = fadd ninf float %y, 0x7FF0000000000000
  %r = frem float 0.0, %n
  ret float %r
}

define float @poison_both(float %x, float %y) {
; CHECK-LABEL: define float @poison_both(
; CHECK-NEXT:    [[NX:%.*]] = fadd ninf float %x, +inf
; CHECK-NEXT:    [[NY:%.*]] = fadd ninf float %y, +inf
; CHECK-NEXT:    [[X:%.*]] = freeze float [[NX]]
; CHECK-NEXT:    [[Y:%.*]] = freeze float [[NY]]
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float [[X]])
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float [[Y]])
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
  %nx = fadd ninf float %x, 0x7FF0000000000000
  %ny = fadd ninf float %y, 0x7FF0000000000000
  %r = frem float %nx, %ny
  ret float %r
}

define float @arguments(float %x, float %y) {
; CHECK-LABEL: define float @arguments(
; CHECK-NEXT:    [[X:%.*]] = freeze float %x
; CHECK-NEXT:    [[Y:%.*]] = freeze float %y
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float [[X]])
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float [[Y]])
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
  %r = frem float %x, %y
  ret float %r
}

define float @defined_arguments(float noundef %x, float noundef %y) {
; CHECK-LABEL: define float @defined_arguments(
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float %y)
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
  %r = frem float %x, %y
  ret float %r
}

define float @frozen_arguments(float %x, float %y) {
; CHECK-LABEL: define float @frozen_arguments(
; CHECK-NEXT:    [[X:%.*]] = freeze float %x
; CHECK-NEXT:    [[Y:%.*]] = freeze float %y
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float [[X]])
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float [[Y]])
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
  %xf = freeze float %x
  %yf = freeze float %y
  %r = frem float %xf, %yf
  ret float %r
}

; These flags can create poison from concrete inputs, even when the
; operands are noundef. Neither the initial nor the loop branches may
; depend on operations carrying the flags.
define float @no_infs(float noundef %x, float noundef %y) {
; CHECK-LABEL: define float @no_infs(
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float %y)
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
; CHECK:       frem.compute:
; CHECK-NEXT:    call { float, i32 } @llvm.frexp.f32.i32(float [[AX]])
; CHECK:         call float @llvm.ldexp.f32.i32(float {{%.*}}, i32 12)
; CHECK:         call { float, i32 } @llvm.frexp.f32.i32(float [[AY]])
; CHECK:         fdiv float 1.000000e+00,
; CHECK:       frem.loop_body:
; CHECK:         fmul float
  %r = frem ninf float %x, %y
  ret float %r
}

define float @no_nans(float noundef %x, float noundef %y) {
; CHECK-LABEL: define float @no_nans(
; CHECK-NEXT:    [[AX:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT:    [[AY:%.*]] = call float @llvm.fabs.f32(float %y)
; CHECK-NEXT:    [[CMP:%.*]] = fcmp ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
; CHECK:       frem.compute:
; CHECK-NEXT:    call { float, i32 } @llvm.frexp.f32.i32(float [[AX]])
; CHECK:         call float @llvm.ldexp.f32.i32(float {{%.*}}, i32 12)
; CHECK:         call { float, i32 } @llvm.frexp.f32.i32(float [[AY]])
; CHECK:         fdiv float 1.000000e+00,
; CHECK:       frem.loop_body:
; CHECK:         fmul float
  %r = frem nnan float %x, %y
  ret float %r
}

; Preserve other applicable flags while removing the poison-generating ones.
define float @other_flags(float %x, float %y) {
; CHECK-LABEL: define float @other_flags(
; CHECK-NEXT:    [[X:%.*]] = freeze float %x
; CHECK-NEXT:    [[Y:%.*]] = freeze float %y
; CHECK-NEXT:    [[AX:%.*]] = call nsz float @llvm.fabs.f32(float [[X]])
; CHECK-NEXT:    [[AY:%.*]] = call nsz float @llvm.fabs.f32(float [[Y]])
; CHECK-NEXT:    [[CMP:%.*]] = fcmp nsz ogt float [[AX]], [[AY]]
; CHECK-NEXT:    br i1 [[CMP]], label %frem.compute, label %frem.else
; CHECK:       frem.compute:
; CHECK-NEXT:    call { float, i32 } @llvm.frexp.f32.i32(float [[AX]])
; CHECK:         call nsz float @llvm.ldexp.f32.i32(float {{%.*}}, i32 12)
; CHECK:         call { float, i32 } @llvm.frexp.f32.i32(float [[AY]])
; CHECK:       frem.loop_body:
; CHECK:         fmul nsz float
  %r = frem nnan ninf nsz float %x, %y
  ret float %r
}

; Discarding the result must not introduce undefined behavior.
define i32 @main() {
; CHECK-LABEL: define i32 @main(
; CHECK-NEXT:    %r = call float @poison_numerator(float 0.000000e+00)
; CHECK-NEXT:    ret i32 0
  %r = call float @poison_numerator(float 0.0)
  ret i32 0
}
