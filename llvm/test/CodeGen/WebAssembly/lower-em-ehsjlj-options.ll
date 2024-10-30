; RUN: llc < %s -enable-emscripten-cxx-exceptions | FileCheck %s --check-prefix=EH
; RUN: llc < %s -enable-emscripten-sjlj | FileCheck %s --check-prefix=SJLJ
; RUN: llc < %s | FileCheck %s --check-prefix=NONE

target triple = "wasm32-unknown-unknown"

; EH: .functype  invoke_vi (i32, i32) -> ()
; EH: .import_module  invoke_vi, env
; EH: .import_name  invoke_vi, invoke_vi
; EH-NOT: .functype  __invoke_void_i32
; EH-NOT: .import_module  __invoke_void_i32
; EH-NOT: .import_name  __invoke_void_i32

; SJLJ: .functype  emscripten_longjmp (i32, i32) -> ()
; SJLJ: .import_module  emscripten_longjmp, env
; SJLJ: .import_name  emscripten_longjmp, emscripten_longjmp
; SJLJ-NOT: .functype  emscripten_longjmp_jmpbuf
; SJLJ-NOT: .import_module  emscripten_longjmp_jmpbuf
; SJLJ-NOT: .import_name  emscripten_longjmp_jmpbuf

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

define void @exception() personality ptr @__gxx_personality_v0 {
; EH-LABEL:   type exception,@function
; NONE-LABEL: type exception,@function
entry:
  invoke void @foo(i32 3)
          to label %invoke.cont unwind label %lpad
; EH:     call invoke_vi
; EH-NOT: call __invoke_void_i32
; NONE:   call foo

invoke.cont:
  invoke void @bar()
          to label %try.cont unwind label %lpad
; EH:     call invoke_v
; EH-NOT: call __invoke_void
; NONE:   call bar

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = call ptr @__cxa_begin_catch(ptr %1) #2
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  ret void
}

define void @setjmp_longjmp() {
; SJLJ-LABEL: type setjmp_longjmp,@function
; NONE-LABEL: type setjmp_longjmp,@function
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  call void @longjmp(ptr %buf, i32 1) #1
  unreachable
; SJLJ: call __wasm_setjmp
; SJLJ: i32.const emscripten_longjmp
; SJLJ-NOT: i32.const emscripten_longjmp_jmpbuf
; SJLJ: call invoke_vii
; SJLJ-NOT: call "__invoke_void_ptr_i32"
; SJLJ: call __wasm_setjmp_test

; NONE: call setjmp
; NONE: call longjmp
}

; Tests whether a user function with 'invoke_' prefix can be used
declare void @invoke_ignoreme()
define void @test_invoke_ignoreme() {
; EH-LABEL:   type test_invoke_ignoreme,@function
; SJLJ-LABEL: type test_invoke_ignoreme,@function
entry:
  call void @invoke_ignoreme()
; EH:   call invoke_ignoreme
; SJLJ: call invoke_ignoreme
  ret void
}

declare void @foo(i32)
declare void @bar()
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
; Function Attrs: returns_twice
declare i32 @setjmp(ptr) #0
; Function Attrs: noreturn
declare void @longjmp(ptr, i32) #1
declare ptr @malloc(i32)
declare void @free(ptr)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
