; RUN: llc -march=hexagon -hexagon-trap-unreachable < %s | FileCheck %s
; CHECK: call abort

define void @fred() #0 {
  unreachable
}

attributes #0 = { nounwind }
