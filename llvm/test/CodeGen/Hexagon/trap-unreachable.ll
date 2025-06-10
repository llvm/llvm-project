; RUN: llc -mtriple=hexagon -trap-unreachable < %s | FileCheck %s

; Trap is implemented via a misaligned load.
; CHECK: memd(##3134984174)

define void @fred() #0 {
  unreachable
}

attributes #0 = { nounwind }
