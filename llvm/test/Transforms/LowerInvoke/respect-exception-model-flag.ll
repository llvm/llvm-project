; Check that the "exception-model" flag is respected. The WebAssembly models
; keep their invokes through isel, so they must survive this pass; every other
; model lowers, since the pass only runs where nothing else will handle them.

; RUN: split-file %s %t
; RUN: opt -passes=lower-invoke -S %t/wasm.ll       | FileCheck --check-prefix=WASM %s
; RUN: opt -passes=lower-invoke -S %t/emscripten.ll | FileCheck --check-prefix=EM %s
; RUN: opt -passes=lower-invoke -S %t/dwarf.ll      | FileCheck --check-prefix=DWARF %s
; RUN: opt -passes=lower-invoke -S %t/none.ll       | FileCheck --check-prefix=NONE %s
; RUN: opt -passes=lower-invoke -S %t/noflag.ll     | FileCheck --check-prefix=NOFLAG %s

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
; EM:         invoke void @g()

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

;--- dwarf.ll
target triple = "x86_64-unknown-linux-gnu"

; DWARF-LABEL: define void @f()
; DWARF-NEXT:  entry:
; DWARF-NEXT:    call void @g()
; DWARF-NOT:     invoke

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
!0 = !{i32 1, !"exception-model", !"dwarf"}

;--- none.ll
target triple = "x86_64-unknown-linux-gnu"

; NONE-LABEL: define void @f()
; NONE-NEXT:  entry:
; NONE-NEXT:    call void @g()
; NONE-NOT:     invoke

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
!0 = !{i32 1, !"exception-model", !"none"}

;--- noflag.ll
target triple = "x86_64-unknown-linux-gnu"

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
