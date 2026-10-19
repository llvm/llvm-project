; RUN: not llc -mtriple=loongarch64 < %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -mtriple=loongarch64 -mattr=+f < %s 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: in function foo void (): rounding mode is not supported by LoongArch hardware

define void @foo() {
entry:
  tail call void @llvm.set.rounding(i32 4)
  ret void
}

declare void @llvm.set.rounding(i32)
