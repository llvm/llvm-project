; RUN: llc < %s -mtriple=ve-unknown-unknown -mattr=+vpu | FileCheck %s

; Scalar argument passing must not change (same tests as in VE/Scalar/callee.ll below - this time with +vpu)

define fastcc i32 @stack_stack_arg_i32_r9(i1 %0, i8 %1, i16 %2, i32 %3, i64 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9) {
; CHECK-LABEL: stack_stack_arg_i32_r9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, 248(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 %9
}

define fastcc i64 @stack_stack_arg_i64_r9(i1 %0, i8 %1, i16 %2, i32 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9) {
; CHECK-LABEL: stack_stack_arg_i64_r9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, 248(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 %9
}

define fastcc float @stack_stack_arg_f32_r9(float %p0, float %p1, float %p2, float %p3, float %p4, float %p5, float %p6, float %p7, float %s0, float %s1) {
; CHECK-LABEL: stack_stack_arg_f32_r9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldu %s0, 252(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
  ret float %s1
}

define fastcc i32 @stack_stack_arg_i32f32_r8(i32 %p0, float %p1, i32 %p2, float %p3, i32 %p4, float %p5, i32 %p6, float %p7, i32 %s0, float %s1) {
; CHECK-LABEL: stack_stack_arg_i32f32_r8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, 240(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 %s0
}

define fastcc float @stack_stack_arg_i32f32_r9(i32 %p0, float %p1, i32 %p2, float %p3, i32 %p4, float %p5, i32 %p6, float %p7, i32 %s0, float %s1) {
; CHECK-LABEL: stack_stack_arg_i32f32_r9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldu %s0, 252(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
  ret float %s1
}

; Vector argument passing (fastcc feature)

; v0-to-v0 passthrough case without vreg copy.
define fastcc <256 x i32> @vreg_arg_v256i32_r0(<256 x i32> %p0) {
; CHECK-LABEL: vreg_arg_v256i32_r0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p0
}

define fastcc <256 x i32> @vreg_arg_v256i32_r1(<256 x i32> %p0, <256 x i32> %p1) {
; CHECK-LABEL: vreg_arg_v256i32_r1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v1
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p1
}

define fastcc <256 x i32> @vreg_arg_v256i32_r2(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2) {
; CHECK-LABEL: vreg_arg_v256i32_r2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v2
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p2
}

define fastcc <256 x i32> @vreg_arg_v256i32_r3(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2, <256 x i32> %p3) {
; CHECK-LABEL: vreg_arg_v256i32_r3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v3
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p3
}

define fastcc <256 x i32> @vreg_arg_v256i32_r4(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2, <256 x i32> %p3, <256 x i32> %p4) {
; CHECK-LABEL: vreg_arg_v256i32_r4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v4
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p4
}

define fastcc <256 x i32> @vreg_arg_v256i32_r5(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2, <256 x i32> %p3, <256 x i32> %p4, <256 x i32> %p5) {
; CHECK-LABEL: vreg_arg_v256i32_r5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v5
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p5
}

define fastcc <256 x i32> @vreg_arg_v256i32_r6(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2, <256 x i32> %p3, <256 x i32> %p4, <256 x i32> %p5, <256 x i32> %p6) {
; CHECK-LABEL: vreg_arg_v256i32_r6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s16, 256
; CHECK-NEXT:    lvl %s16
; CHECK-NEXT:    vor %v0, (0)1, %v6
; CHECK-NEXT:    b.l.t (, %s10)
  ret <256 x i32> %p6
}

; TODO: Uncomment test when vector loads are upstream (vreg stack passing).
; define <256 x i32> @vreg_arg_v256i32_r7(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2, <256 x i32> %p3, <256 x i32> %p4, <256 x i32> %p5, <256 x i32> %p6, <256 x i32> %p7) {
;   ret <256 x i32> %p7
; }

; define <256 x i32> @vreg_arg_v256i32_r8(<256 x i32> %p0, <256 x i32> %p1, <256 x i32> %p2, <256 x i32> %p3, <256 x i32> %p4, <256 x i32> %p5, <256 x i32> %p6, <256 x i32> %p7, <256 x i32> %p8) {
;   ret <256 x i32> %p8
; }
