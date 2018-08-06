; RUN: llc -mtriple=thumbv7 -trap-unreachable < %s | FileCheck %s
; CHECK: .inst.n 0xdefe

define void @test() #0 {
  unreachable
}

attributes #0 = { nounwind }
