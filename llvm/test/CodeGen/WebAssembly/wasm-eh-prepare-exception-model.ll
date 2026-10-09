; The pass should only run when the module selects the Wasm model.

; RUN: split-file %s %t
; RUN: opt -passes=wasm-eh-prepare -S %t/wasm.ll   | FileCheck %s --check-prefix=WASM
; RUN: opt -passes=wasm-eh-prepare -S %t/noflag.ll | FileCheck %s --check-prefix=NOFLAG

;--- wasm.ll
target triple = "wasm32-unknown-unknown"

declare void @g()
declare i32 @__gxx_wasm_personality_v0(...)

; WASM-LABEL: catch.start:
; WASM-NEXT:    %cp = catchpad within %cs [ptr null]
; WASM-NEXT:    %exn = call ptr @llvm.wasm.catch(i32 0)
; WASM-NEXT:    catchret

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

;--- noflag.ll
target triple = "wasm32-unknown-unknown"

declare void @g()
declare i32 @__gxx_wasm_personality_v0(...)

; NOFLAG-LABEL: catch.start:
; NOFLAG-NEXT:    %cp = catchpad within %cs [ptr null]
; NOFLAG-NEXT:    %e = call ptr @llvm.wasm.get.exception(token %cp)
; NOFLAG-NEXT:    %s = call i32 @llvm.wasm.get.ehselector(token %cp)
; NOFLAG-NOT:   llvm.wasm.catch

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
