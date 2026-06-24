; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm64-apple-ios14.0"

; Test: Function with nooutline attribute should NOT be split
; CHECK-LABEL: define {{.*}}@func_with_nooutline_attr
; CHECK-NOT: func_with_nooutline_attr{{.*}}.cold
define void @func_with_nooutline_attr(i1 %cond) "nooutline" {
entry:
  br i1 %cond, label %fast_path, label %cold_path

fast_path:
  ret void

cold_path:
  call void @cold_helper()
  call void @cold_helper()
  call void @cold_helper()
  ret void
}

declare void @cold_helper() cold
