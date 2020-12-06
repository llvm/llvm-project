; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; Intrinsic call to @llvm.assume should not prevent tail call optimization.
; CHECK-LABEL: foo:
; CHECK:       jmp bar # TAILCALL
define i8* @foo() {
  %1 = tail call i8* @bar()
  %2 = icmp ne i8* %1, null
  tail call void @llvm.assume(i1 %2)
  ret i8* %1
}

declare dso_local i8* @bar()
declare void @llvm.assume(i1)

