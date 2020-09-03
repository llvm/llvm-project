; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i32 @stack_stack_arg_i32_r9(i1 %0, i8 %1, i16 %2, i32 %3, i64 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9) {
; CHECK-LABEL: stack_stack_arg_i32_r9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, 424(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 %9
}

define i64 @stack_stack_arg_i64_r9(i1 %0, i8 %1, i16 %2, i32 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9) {
; CHECK-LABEL: stack_stack_arg_i64_r9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, 424(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 %9
}

define float @stack_stack_arg_f32_r9(float %p0, float %p1, float %p2, float %p3, float %p4, float %p5, float %p6, float %p7, float %s0, float %s1) {
; CHECK-LABEL: stack_stack_arg_f32_r9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, 428(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  ret float %s1
}

define i32 @stack_stack_arg_i32f32_r8(i32 %p0, float %p1, i32 %p2, float %p3, i32 %p4, float %p5, i32 %p6, float %p7, i32 %s0, float %s1) {
; CHECK-LABEL: stack_stack_arg_i32f32_r8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldl.sx %s0, 416(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 %s0
}

define float @stack_stack_arg_i32f32_r9(i32 %p0, float %p1, i32 %p2, float %p3, i32 %p4, float %p5, i32 %p6, float %p7, i32 %s0, float %s1) {
; CHECK-LABEL: stack_stack_arg_i32f32_r9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, 428(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  ret float %s1
}
