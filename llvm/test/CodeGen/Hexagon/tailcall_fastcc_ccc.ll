; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

declare hidden fastcc void @callee(i32, i32) #0
declare hidden void @callee2(i32, i32) #0

; CHECK: jump callee
define void @caller(i32 %pp) #0 {
entry:
  tail call fastcc void @callee(i32 %pp, i32 0)
  ret void
}

; CHECK: jump callee2
define void @caller2(i32 %pp) #0 {
entry:
  tail call fastcc void @callee2(i32 %pp, i32 0)
  ret void
}

attributes #0 = { nounwind }
