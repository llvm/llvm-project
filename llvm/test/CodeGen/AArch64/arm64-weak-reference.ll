; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s

@x = extern_weak global i32

define i32 @fn() nounwind ssp {
; CHECK-LABEL: fn:
; CHECK: .weak_reference
  %val = load i32, ptr @x, align 4
  ret i32 %val
}
