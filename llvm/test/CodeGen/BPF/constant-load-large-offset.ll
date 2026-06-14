; Regression test for https://github.com/llvm/llvm-project/issues/187756
; RUN: llc -mtriple=bpf < %s | FileCheck %s
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

%elem = type { [3735923519 x i8], i8, [1152921500870923457 x i8] }

@g = constant [16 x %elem] [%elem { [3735923519 x i8] zeroinitializer, i8 42, [1152921500870923457 x i8] zeroinitializer }, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer, %elem zeroinitializer]

define i8 @test() {
entry:
  %v = load i8, ptr @g, align 1
  ret i8 %v
; CHECK-LABEL: test:
; CHECK: r1 = g
; CHECK: w0 = *(u8 *)(r1 + 0)
}
