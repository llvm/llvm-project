; RUN: opt -mtriple=aarch64-none-elf -mattr=+sve2 -O2 -disable-output

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64--linux-gnueabihf"

; This verifies LAA does not attempt to get a fixed element count on a scalable vector.
; From issue: https://github.com/llvm/llvm-project/issues/153797

define i32 @gradient_fast_par_for_gradient_fast_s0_x_v18_v22(ptr %gradient_fast, i64 %0, ptr %1) {
entry:
  br label %"2_for_gradient_fast.s0.x.v20.v23"

"2_for_gradient_fast.s0.x.v20.v23":               ; preds = %"2_for_gradient_fast.s0.x.v20.v23", %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %"2_for_gradient_fast.s0.x.v20.v23" ]
  %2 = shl i64 %indvars.iv, 1
  %3 = add i64 %2, %0
  %4 = trunc i64 %indvars.iv to i32
  %5 = insertelement <vscale x 4 x i32> zeroinitializer, i32 %4, i64 0
  %6 = getelementptr i32, ptr %gradient_fast, i64 %3
  store <vscale x 4 x i32> %5, ptr %6, align 4
  %.reass3 = or i32 %4, 1
  %7 = insertelement <vscale x 4 x i32> zeroinitializer, i32 %.reass3, i64 0
  %8 = shufflevector <vscale x 4 x i32> %7, <vscale x 4 x i32> zeroinitializer, <vscale x 4 x i32> zeroinitializer
  %9 = getelementptr i32, ptr %1, i64 %3
  store <vscale x 4 x i32> %8, ptr %9, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %.not = icmp eq i64 %indvars.iv, 16
  br i1 %.not, label %"2_end_for_gradient_fast.s0.x.v20.v23", label %"2_for_gradient_fast.s0.x.v20.v23"

"2_end_for_gradient_fast.s0.x.v20.v23":           ; preds = %"2_for_gradient_fast.s0.x.v20.v23"
  ret i32 0
}
