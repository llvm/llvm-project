; The "exception-model" module flag selects the EH model with no
; -exception-model on the command line. Wasm triples default to no exception
; handling, so a module without the flag keeps neither lowering.

; RUN: split-file %s %t

; RUN: llc -wasm-use-legacy-eh=false -mattr=+exception-handling %t/wasm.ll -o - | FileCheck %s --check-prefix=WASM
; RUN: llc %t/emscripten.ll -o - | FileCheck %s --check-prefix=EM
; RUN: llc -wasm-use-legacy-eh=false -mattr=+exception-handling %t/noflag.ll -o - | FileCheck %s --check-prefix=NOFLAG

; Without the flag, -exception-model still selects the model.

; RUN: llc -wasm-use-legacy-eh=false -mattr=+exception-handling -exception-model=wasm %t/noflag.ll -o - | FileCheck %s --check-prefix=WASM
; RUN: llc -exception-model=emscripten %t/noflag-em.ll -o - | FileCheck %s --check-prefix=EM

; WASM: .tagtype __cpp_exception i32
; WASM: try_table  (catch __cpp_exception 0)

; EM: .functype invoke_v (i32) -> ()
; EM: call invoke_v
; EM: call __cxa_find_matching_catch_3

; NOFLAG: call g
; NOFLAG-NOT: try_table
; NOFLAG-NOT: invoke_v

;--- wasm.ll
target triple = "wasm32-unknown-unknown"

declare void @g()
declare i32 @__gxx_wasm_personality_v0(...)
declare void @__cxa_end_catch()

define void @f() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %catch.dispatch
catch.dispatch:
  %cs = catchswitch within none [label %catch.start] unwind to caller
catch.start:
  %cp = catchpad within %cs [ptr null]
  %e = call ptr @llvm.wasm.get.exception(token %cp)
  %s = call i32 @llvm.wasm.get.ehselector(token %cp)
  call void @__cxa_end_catch() [ "funclet"(token %cp) ]
  catchret from %cp to label %cont
cont:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}

;--- emscripten.ll
target triple = "wasm32-unknown-emscripten"

@_ZTIi = external constant ptr

declare void @g()
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define void @f() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %lpad
lpad:
  %p = landingpad { ptr, i32 } catch ptr @_ZTIi
  %e = extractvalue { ptr, i32 } %p, 0
  %q = call ptr @__cxa_begin_catch(ptr %e)
  call void @__cxa_end_catch()
  br label %cont
cont:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"emscripten"}

;--- noflag.ll
target triple = "wasm32-unknown-unknown"

declare void @g()
declare i32 @__gxx_wasm_personality_v0(...)
declare void @__cxa_end_catch()

define void @f() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %catch.dispatch
catch.dispatch:
  %cs = catchswitch within none [label %catch.start] unwind to caller
catch.start:
  %cp = catchpad within %cs [ptr null]
  %e = call ptr @llvm.wasm.get.exception(token %cp)
  %s = call i32 @llvm.wasm.get.ehselector(token %cp)
  call void @__cxa_end_catch() [ "funclet"(token %cp) ]
  catchret from %cp to label %cont
cont:
  ret void
}

;--- noflag-em.ll
target triple = "wasm32-unknown-emscripten"

@_ZTIi = external constant ptr

declare void @g()
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define void @f() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @g() to label %cont unwind label %lpad
lpad:
  %p = landingpad { ptr, i32 } catch ptr @_ZTIi
  %e = extractvalue { ptr, i32 } %p, 0
  %q = call ptr @__cxa_begin_catch(ptr %e)
  call void @__cxa_end_catch()
  br label %cont
cont:
  ret void
}
