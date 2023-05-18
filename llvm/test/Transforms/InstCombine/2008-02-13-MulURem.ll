; RUN: opt < %s -passes=instcombine -S | FileCheck %s
; PR1933

; CHECK: rem

define i32 @fold(i32 %a) {
  %s = mul i32 %a, 3
  %c = urem i32 %s, 3
  ret i32 %c
}
