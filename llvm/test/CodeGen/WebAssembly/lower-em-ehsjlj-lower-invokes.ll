; The pass lowers the invokes the wasm models do not keep: Wasm EH keeps them
; for WasmEHPrepare and Emscripten EH rewrites them into __invoke_* calls.

; RUN: split-file %s %t

; RUN: opt -mtriple=wasm32-unknown-unknown -passes=wasm-lower-em-ehsjlj -S %t/wasm.ll | FileCheck --check-prefix=WASM %s
; RUN: opt -mtriple=wasm32-unknown-emscripten -passes=wasm-lower-em-ehsjlj -S %t/emscripten.ll | FileCheck --check-prefix=EM %s
; RUN: opt -mtriple=wasm32-unknown-unknown -passes=wasm-lower-em-ehsjlj -S %t/noflag.ll | FileCheck --check-prefix=NOFLAG %s

;--- wasm.ll
target triple = "wasm32-unknown-unknown"

; WASM-LABEL: define void @f()
; WASM:         invoke void @g()

define void @f() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %lpad

lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p

cont:
  ret void
}

declare void @g()
declare i32 @__gxx_wasm_personality_v0(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}

;--- emscripten.ll
target triple = "wasm32-unknown-emscripten"

; EM-LABEL: define void @f()
; EM:         call void @__invoke_void(ptr @g)
; EM-NOT:     invoke void @g()

define void @f() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %lpad

lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p

cont:
  ret void
}

declare void @g()
declare i32 @__gxx_personality_v0(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"emscripten"}

;--- noflag.ll
target triple = "wasm32-unknown-unknown"

; NOFLAG-LABEL: define void @f()
; NOFLAG-NEXT:  entry:
; NOFLAG-NEXT:    call void @g()
; NOFLAG-NOT:     invoke

define void @f() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %lpad

lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p

cont:
  ret void
}

declare void @g()
declare i32 @__gxx_personality_v0(...)
