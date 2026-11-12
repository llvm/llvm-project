; RUN: opt -S < %s -passes=objc-arc-expand,objc-arc-contract | FileCheck %s
; Don't crash.  Reproducer for a use_iterator bug from r203364.
; rdar://problem/16333235
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin13.2.0"

%struct = type { ptr, ptr }

; CHECK-LABEL: @foo() {
define internal ptr @foo() {
entry:
  %call = call ptr @bar()
; CHECK: %retained1 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call)
  %retained1 = call ptr @llvm.objc.retain(ptr %call)
  %isnull = icmp eq ptr %retained1, null
  br i1 %isnull, label %cleanup, label %if.end

if.end:
; CHECK: %retained2 = call ptr @llvm.objc.retain(ptr %retained1)
  %retained2 = call ptr @llvm.objc.retain(ptr %retained1)
  br label %cleanup

cleanup:
  %retval = phi ptr [ %retained2, %if.end ], [ null, %entry ]
  ret ptr %retval
}

declare ptr @bar()

declare extern_weak ptr @llvm.objc.retain(ptr)
