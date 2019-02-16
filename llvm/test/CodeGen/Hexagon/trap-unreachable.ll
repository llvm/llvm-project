; RUN: llc -march=hexagon -trap-unreachable < %s | FileCheck %s
; CHECK: trap

define void @fred() #0 {
  unreachable
}

attributes #0 = { nounwind }
