; RUN: llc -mtriple=hexagon -trap-unreachable < %s | FileCheck %s

; Trap is the word 0x00110011, a duplex where both slots write R1.
; CHECK: .word 1114129

define void @fred() #0 {
  unreachable
}

attributes #0 = { nounwind }
