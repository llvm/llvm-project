; RUN: llc -march=hexagon < %s | FileCheck %s

; Don't output falign for function entries when optimizing for size.
; CHECK-NOT: falign

define i32 @f0() #0 {
b0:
  ret i32 0
}

attributes #0 = { optsize }
