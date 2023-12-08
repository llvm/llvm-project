; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure that the first formal argument is not loaded from memory.
; CHECK-NOT: memw

define i32 @fred(i32 %a0, ...) #0 {
b1:
  %v2 = add i32 %a0, 1
  ret i32 %v2
}

attributes #0 = { nounwind }
