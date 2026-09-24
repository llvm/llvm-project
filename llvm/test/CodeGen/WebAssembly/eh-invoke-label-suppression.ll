; RUN: llc -exception-model=wasm -mattr=+exception-handling -wasm-use-legacy-eh=false < %s | FileCheck %s

; Wasm exception tables don't reference the per-invoke try-range
; labels, so they aren't emitted: the call sits directly inside
; try_table. The landing-pad entry label is still emitted.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: test0:
; CHECK:        try_table
; CHECK-NEXT:     call foo
; CHECK-NEXT:     br
; CHECK:      .LBB0_{{[0-9]+}}:
; CHECK:        # EH_LABEL
define void @test0() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %catch.start] unwind to caller

catch.start:
  %pad = catchpad within %cs [ptr null]
  %exn = call ptr @llvm.wasm.get.exception(token %pad)
  %sel = call i32 @llvm.wasm.get.ehselector(token %pad)
  %obj = call ptr @__cxa_begin_catch(ptr %exn) [ "funclet"(token %pad) ]
  call void @__cxa_end_catch() [ "funclet"(token %pad) ]
  catchret from %pad to label %try.cont

try.cont:
  ret void
}

declare void @foo()
declare i32 @__gxx_wasm_personality_v0(...)
declare ptr @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
