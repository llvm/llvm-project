; RUN: not --crash llc < %s -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Tests if the compiler correctly errors out when a function having EH pads does
; not have a correct Wasm personality function.

define void @test() personality ptr @invalid_personality {
; CHECK: LLVM ERROR: Function 'test' does not have a correct Wasm personality function '__gxx_wasm_personality_v0'
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch, %entry
  ret void
}

declare void @foo()
declare i32 @invalid_personality(...)
