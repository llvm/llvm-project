; RUN: opt -S -passes=no-op-module -instnamer-after-each-pass %s | FileCheck %s

; Name unnamed values in both functions. Local names may repeat between them.
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

; CHECK-LABEL: define i32 @first(
; CHECK-SAME: i32 %[[ARG1:arg\.[0-9]+]])
; CHECK: %[[I1:i\.[0-9]+]] = add i32 %[[ARG1]], 1
; CHECK-LABEL: define i32 @second(
; CHECK-SAME: i32 %[[ARG2:arg\.[0-9]+]])
; CHECK: %[[I2:i\.[0-9]+]] = add i32 %[[ARG2]], 2
