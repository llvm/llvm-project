; RUN: not --crash llc -O0 < %s -mtriple=avr 2>&1 | FileCheck %s

define void @foo() {
entry:
; CHECK: Invalid register name "r28".
  %val1 = call i8 @llvm.read_register.i8(metadata !0)
  ret void
}

declare i8 @llvm.read_register.i8(metadata)

!0 = !{!"r28"}
