; RUN: opt < %s -cost-model -analyze | FileCheck %s

; The cost model does not have any target information so it can't make a decision.
; Notice that OPT does not read the triple information from the module itself, only through the command line.

; This info ignored:
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: Unknown cost {{.*}} add
;CHECK: Unknown cost {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i32 %arg, %arg
  ret i32 %e
}
