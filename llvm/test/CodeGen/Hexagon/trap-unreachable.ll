; RUN: llc -mtriple=hexagon -trap-unreachable < %s | FileCheck %s

; Trap is a 32-bit zero word that the processor decodes as an illegal duplex.
; CHECK: .word 0

define void @fred() #0 {
  unreachable
}

attributes #0 = { nounwind }
