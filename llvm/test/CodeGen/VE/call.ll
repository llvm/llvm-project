; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

declare i32 @sample_add(i32, i32)
declare i32 @stack_callee_int(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
declare i32 @stack_callee_int_szext(i1 signext, i8 zeroext, i32, i32, i32, i32, i32, i32, i16 zeroext, i8 signext)
declare float @stack_callee_float(float, float, float, float, float, float, float, float, float, float)
declare void @test(i64)

define i32 @sample_call() {
; CHECK-LABEL: sample_call:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, sample_add@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, sample_add@hi(, %s0)
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    or %s1, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i32 @sample_add(i32 1, i32 2)
  ret i32 %r
}

define i32 @stack_call_int() {
; CHECK-LABEL: stack_call_int:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 10, (0)1
; CHECK-NEXT:    st %s0, 248(, %s11)
; CHECK-NEXT:    or %s34, 9, (0)1
; CHECK-NEXT:    lea %s0, stack_callee_int@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, stack_callee_int@hi(, %s0)
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    or %s1, 2, (0)1
; CHECK-NEXT:    or %s2, 3, (0)1
; CHECK-NEXT:    or %s3, 4, (0)1
; CHECK-NEXT:    or %s4, 5, (0)1
; CHECK-NEXT:    or %s5, 6, (0)1
; CHECK-NEXT:    or %s6, 7, (0)1
; CHECK-NEXT:    or %s7, 8, (0)1
; CHECK-NEXT:    st %s34, 240(, %s11)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i32 @stack_callee_int(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10)
  ret i32 %r
}

define i32 @stack_call_int_szext() {
; CHECK-LABEL: stack_call_int_szext:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, -1, (0)1
; CHECK-NEXT:    st %s0, 248(, %s11)
; CHECK-NEXT:    lea %s34, 65535
; CHECK-NEXT:    lea %s1, stack_callee_int_szext@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, stack_callee_int_szext@hi(, %s1)
; CHECK-NEXT:    lea %s1, 255
; CHECK-NEXT:    or %s2, 3, (0)1
; CHECK-NEXT:    or %s3, 4, (0)1
; CHECK-NEXT:    or %s4, 5, (0)1
; CHECK-NEXT:    or %s5, 6, (0)1
; CHECK-NEXT:    or %s6, 7, (0)1
; CHECK-NEXT:    or %s7, 8, (0)1
; CHECK-NEXT:    st %s34, 240(, %s11)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i32 @stack_callee_int_szext(i1 -1, i8 -1, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i16 -1, i8 -1)
  ret i32 %r
}

define float @stack_call_float() {
; CHECK-LABEL: stack_call_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 1092616192
; CHECK-NEXT:    st %s0, 248(, %s11)
; CHECK-NEXT:    lea.sl %s34, 1091567616
; CHECK-NEXT:    lea.sl %s0, 1065353216
; CHECK-NEXT:    lea.sl %s1, 1073741824
; CHECK-NEXT:    lea.sl %s2, 1077936128
; CHECK-NEXT:    lea.sl %s3, 1082130432
; CHECK-NEXT:    lea.sl %s4, 1084227584
; CHECK-NEXT:    lea.sl %s5, 1086324736
; CHECK-NEXT:    lea.sl %s6, 1088421888
; CHECK-NEXT:    lea.sl %s7, 1090519040
; CHECK-NEXT:    lea %s35, stack_callee_float@lo
; CHECK-NEXT:    and %s35, %s35, (32)0
; CHECK-NEXT:    lea.sl %s12, stack_callee_float@hi(, %s35)
; CHECK-NEXT:    st %s34, 240(, %s11)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call float @stack_callee_float(float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0)
  ret float %r
}

define float @stack_call_float2(float %p0) {
; CHECK-LABEL: stack_call_float2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s0, 248(, %s11)
; CHECK-NEXT:    lea %s1, stack_callee_float@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, stack_callee_float@hi(, %s1)
; CHECK-NEXT:    st %s0, 240(, %s11)
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    or %s3, 0, %s0
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    or %s5, 0, %s0
; CHECK-NEXT:    or %s6, 0, %s0
; CHECK-NEXT:    or %s7, 0, %s0
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call float @stack_callee_float(float %p0, float %p0, float %p0, float %p0, float %p0, float %p0, float %p0, float %p0, float %p0, float %p0)
  ret float %r
}

