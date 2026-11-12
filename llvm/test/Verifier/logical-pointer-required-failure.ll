; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

define void @gep_with_logical_required(ptr %src, i32 %index) {
entry:
; CHECK: Non-logical getelementptr disallowed for this module.
  %ptr = getelementptr i8, ptr %src, i32 0
  ret void
}

define void @alloca_with_logical_required() {
entry:
; CHECK: Non-logical alloca disallowed for this module.
  %tmp = alloca i32
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"require-logical-pointer", i32 1}
