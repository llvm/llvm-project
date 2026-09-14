; RUN: llc -mtriple=hexagon -trap-unreachable < %s | FileCheck %s

; Trap is the word 0x9b810001: R1 = memw(R1++#0) with both outputs writing R1.
; CHECK: .word 2608922625

define void @fred() #0 {
  unreachable
}

attributes #0 = { nounwind }
