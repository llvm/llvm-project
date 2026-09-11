; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: unknown floating-point class: normal

define i1 @parse_is_fpclass_1(float %x) {
  %1 = tail call i1 @llvm.is.fpclass.f32(float %x, i32 fc"zero normal")
  ret i1 %1
}
