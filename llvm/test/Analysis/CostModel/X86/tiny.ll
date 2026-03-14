; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: cost of 1 {{.*}} add
;CHECK: cost of 0 {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i32 %arg, %arg
  ret i32 %e
}
