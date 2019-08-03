; RUN: llc -mtriple=x86_64-unknown-unknown -no-integrated-as < %s 2>&1 | FileCheck %s

@x = global i32 0, align 4

define void @foo() {
; CHECK-LABEL: foo:
  call void asm sideeffect "foo $0", "n"(i32 42) nounwind
; CHECK:      #APP
; CHECK-NEXT: foo    $42
; CHECK-NEXT: #NO_APP
  ret void
; CHECK-NEXT: retq
}
