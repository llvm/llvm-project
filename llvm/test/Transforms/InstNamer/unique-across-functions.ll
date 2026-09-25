; RUN: opt -S -passes=no-op-module -instnamer-after-each-pass %s | FileCheck %s

; An unnamed value in another function must not receive a name already used
; in this pipeline.
define i32 @first(i32) {
entry:
  %1 = add i32 %0, 1
  ret i32 %1
}

define i32 @second(i32) {
entry:
  %1 = add i32 %0, 2
  ret i32 %1
}

; CHECK-LABEL: define i32 @first(i32 %arg.0)
; CHECK: %i.1 = add i32 %arg.0, 1
; CHECK-LABEL: define i32 @second(i32 %arg.2)
; CHECK: %i.3 = add i32 %arg.2, 2
