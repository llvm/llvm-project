; RUN: not llc < %s -mtriple=riscv32 2>&1 | FileCheck %s

define i32 @get_invalid_reg() nounwind {
entry:
; CHECK: error: <unknown>:0:0: invalid register "notareg" for llvm.read_register
  %reg = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %reg
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = !{!"notareg\00"}
