; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attribute 'dead_on_return' applied to incompatible type!
; CHECK-NEXT: ptr @arg_not_pointer
define void @arg_not_pointer(i32 dead_on_return %arg) {
  ret void
}
