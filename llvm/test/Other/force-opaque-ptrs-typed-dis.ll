; RUN: llvm-as --force-opaque-pointers < %s | llvm-dis | FileCheck %s

; CHECK: @g = external global i16
@g = external global i16

define void @f(i32* %p) {
; CHECK-LABEL: @f(
; CHECK-NEXT:    [[A:%.*]] = alloca i17, align 4
; CHECK-NEXT:    ret void
;
  %a = alloca i17
  ret void
}
