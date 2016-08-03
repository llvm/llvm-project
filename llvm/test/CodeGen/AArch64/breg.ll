; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s

@stored_label = global i8* null

define void @foo() {
; CHECK-LABEL: foo:
  %lab = load i8*, i8** @stored_label
  indirectbr i8* %lab, [label  %otherlab, label %retlab]
; CHECK: adrp {{x[0-9]+}}, stored_label
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:stored_label]
; CHECK: br {{x[0-9]+}}

otherlab:
  ret void
retlab:
  ret void
}
