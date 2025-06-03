; Make sure MSan doesn't insert shadow checks for @llvm.is.constant.* arguments.

; RUN: opt < %s -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck -check-prefixes=CHECK %s
; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck -check-prefixes=CHECK %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define dso_local i32 @bar(i32 %v) local_unnamed_addr sanitize_memory {
entry:
  %0 = tail call i1 @llvm.is.constant.i32(i32 %v)
  %1 = zext i1 %0 to i32
  ret i32 %1
}

; CHECK-LABEL: bar
; CHECK-NOT: call void @__msan_warning_with_origin

; Function Attrs: nounwind readnone
declare i1 @llvm.is.constant.i32(i32)
