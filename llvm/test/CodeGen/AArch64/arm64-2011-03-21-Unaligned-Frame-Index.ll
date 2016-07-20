; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s
define void @foo(i64 %val) {
; CHECK: foo
;   The stack frame store is not 64-bit aligned. Make sure we use an
;   instruction that can handle that.
; CHECK: stur x0, [sp, #20]
  %a = alloca [49 x i32], align 4
  %p32 = getelementptr inbounds [49 x i32], [49 x i32]* %a, i64 0, i64 2
  %p = bitcast i32* %p32 to i64*
  store i64 %val, i64* %p, align 8
  ret void
}
