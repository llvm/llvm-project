; NOTE: This test verifies that CodeGenPrepare doesn't crash when PSI is null
; RUN: opt -passes=codegenprepare < %s -S | FileCheck %s
;
; This test case triggered a null pointer dereference in CodeGenPrepare::_run()
; when ProfileSummaryInfo (PSI) was not available. The pass attempted to call
; PSI->hasHugeWorkingSetSize() without checking if PSI was null.
;
; The fix adds null checks before dereferencing PSI.
; See: https://github.com/llvm/llvm-project/issues/173360

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %iv.next.reg2mem = alloca i32, align 4
; CHECK-NEXT:    ret void
define void @f(ptr %A, i32 %n) {
entry:
  %iv.next.reg2mem = alloca i32, align 4
  ret void
}
