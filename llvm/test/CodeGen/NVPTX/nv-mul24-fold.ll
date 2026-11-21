; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nounwind readone
declare i32 @llvm.nvvm.mul24.ui(i32, i32)
declare i32 @llvm.nvvm.mul24.i(i32, i32)

; Function Attrs: alwaysinline nounwind
define internal i32 @_test_mul24_i(i32 %a) {
entry:
  ; CHECK-NOT: call i32 @llvm.nvvm.mul24.i
  ; CHECK: [[Folded:%[0-9]+]] = add i32 %a, -30
  ; 16777206 is -10 for a 24 bit signed integer.
  %1 = tail call i32 @llvm.nvvm.mul24.i(i32 16777206, i32 3)
  %2 = add i32 %1, %a
  ret i32 %2
}

; Function Attrs: alwaysinline nounwind
define internal i32 @_test_mul24_ui(i32 %a) {
entry:
  ; CHECK-NOT: call i32 @llvm.nvvm.mul24.ui
  ; CHECK: [[Folded:%[0-9]+]] = add i32 %a, 18750
  %1 = tail call i32 @llvm.nvvm.mul24.ui(i32 150, i32 125)
  %2 = add i32 %1, %a
  ret i32 %2
}
