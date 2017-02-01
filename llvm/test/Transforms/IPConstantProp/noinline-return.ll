; RUN: opt %s -ipsccp -S | FileCheck %s

define i32 @tinkywinky() #0 {
entry:
  ret i32 5
}

define i32 @patatino() {
entry:
  %call = call i32 @tinkywinky()

; Check that we don't propagate the return value of
; @tinkywinky.
; CHECK: call i32 @dipsy(i32 %call)
  %call1 = call i32 @dipsy(i32 %call)
  ret i32 %call1
}

declare i32 @dipsy(i32)

attributes #0 = { noinline }
