; Without Wasm or Emscripten exception handling the pass lowers invokes to
; calls. Wasm EH keeps them, and Emscripten EH turns them into __invoke_* calls.

; RUN: split-file %s %t

; RUN: opt -passes=wasm-lower-em-ehsjlj -S %t/wasm.ll | FileCheck --check-prefix=WASM %s
; RUN: opt -passes=wasm-lower-em-ehsjlj -S %t/emscripten.ll | FileCheck --check-prefix=EM %s
; RUN: opt -passes=wasm-lower-em-ehsjlj -S %t/noflag.ll | FileCheck --check-prefix=NOFLAG %s

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
; EM:         call wasm_emscripten_invokecc void @__invoke_void(ptr @g)
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

; The landing pad is unreachable once the invoke is lowered, so it is removed.
; NOFLAG-LABEL: define void @f()
; NOFLAG-NEXT:  entry:
; NOFLAG-NEXT:    call void @g()
; NOFLAG-NEXT:    br label %cont
; NOFLAG-EMPTY:
; NOFLAG-NEXT:  cont:
; NOFLAG-NEXT:    ret void
; NOFLAG-NEXT:  }

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
