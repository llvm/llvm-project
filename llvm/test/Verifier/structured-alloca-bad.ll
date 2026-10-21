; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

define void @missing_attribute() {
entry:
; CHECK: @llvm.structured.alloca calls require elementtype attribute.
  %ptr = call ptr @llvm.structured.alloca()
  ret void
}
