; RUN: llc -mcpu=pwr9 -mtriple=powerpc-ibm-aix-xcoff -verify-machineinstrs < %s | FileCheck %s


define dso_local signext i32 @foo() {
entry:
  ret i32 55
; CHECK-LABEL: .foo:
; CHECK: li 3, 55
; CHECK: blr
}

