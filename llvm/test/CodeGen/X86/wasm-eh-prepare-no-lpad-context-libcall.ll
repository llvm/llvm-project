; RUN: not opt -passes='require<runtime-libcall-info>,wasm-eh-prepare' -disable-output %s 2>&1 | FileCheck %s

; _Unwind_GetWasmLPadContext is only available on wasm targets.

; CHECK: error: no libcall available for wasm landing pad context

target triple = "x86_64-unknown-linux-gnu"

declare void @g()
declare i32 @__gxx_wasm_personality_v0(...)

define void @f() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %catch.dispatch
catch.dispatch:
  %cs = catchswitch within none [label %catch.start] unwind to caller
catch.start:
  %cp = catchpad within %cs [ptr null]
  %e = call ptr @llvm.wasm.get.exception(token %cp)
  %s = call i32 @llvm.wasm.get.ehselector(token %cp)
  catchret from %cp to label %cont
cont:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}
