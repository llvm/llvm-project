; RUN: opt -rewrite-statepoints-for-gc -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare cc42 double @llvm.experimental.deoptimize.f64(...)

define double @caller_3() gc "statepoint-example" {
; CHECK-LABELL @caller_3(
; CHECK: call cc42 token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint
; CHECK:  unreachable

entry:
  %val = call cc42 double(...) @llvm.experimental.deoptimize.f64() [ "deopt"() ]
  ret double %val
}
