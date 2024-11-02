; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; Intrinsic call to @llvm.assume should not prevent tail call optimization.
; CHECK-LABEL: foo:
; CHECK:       jmp bar # TAILCALL
define ptr @foo() {
  %1 = tail call ptr @bar()
  %2 = icmp ne ptr %1, null
  tail call void @llvm.assume(i1 %2)
  ret ptr %1
}

declare dso_local ptr @bar()
declare void @llvm.assume(i1)

