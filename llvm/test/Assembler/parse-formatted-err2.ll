; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: class specifications must be disjoint: finite

define i1 @parse_is_fpclass_1(float %x) {
  %1 = tail call i1 @llvm.is.fpclass.f32(float %x, i32 fc"zero finite")
  ret i1 %1
}
