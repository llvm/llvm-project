; RUN: llc < %s | FileCheck %s
; Make sure we don't crash in AArch64RedundantCopyElimination when a
; MachineBasicBlock is empty.  PR29035.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i8* @bar()

; CHECK-LABEL: foo:
; CHECK: tbz
; CHECK: mov{{.*}}, #1
; CHECK: ret
; CHECK: bl bar
; CHECK: cbnz
; CHECK: ret
define i1 @foo(i1 %start) {
entry:
  br i1 %start, label %cleanup, label %if.end

if.end:                                           ; preds = %if.end, %entry
  %call = tail call i8* @bar()
  %cmp = icmp eq i8* %call, null
  br i1 %cmp, label %cleanup, label %if.end

cleanup:                                          ; preds = %if.end, %entry
  %retval.0 = phi i1 [ true, %entry ], [ false, %if.end ]
  ret i1 %retval.0
}
