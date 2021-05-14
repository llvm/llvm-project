; RUN: llc < %s -enable-emscripten-cxx-exceptions | FileCheck %s --check-prefix=EH
; RUN: llc < %s -enable-emscripten-sjlj | FileCheck %s --check-prefix=SJLJ
; RUN: llc < %s | FileCheck %s --check-prefix=NONE
; RUN: not --crash llc < %s -enable-emscripten-cxx-exceptions -exception-model=wasm 2>&1 | FileCheck %s --check-prefix=WASM-EH-EM-EH

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

define void @exception() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
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
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  %3 = call i8* @__cxa_begin_catch(i8* %1) #2
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
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable
; SJLJ: call saveSetjmp
; SJLJ: i32.const emscripten_longjmp
; SJLJ-NOT: i32.const emscripten_longjmp_jmpbuf
; SJLJ: call invoke_vii
; SJLJ-NOT: call "__invoke_void_%struct.__jmp_buf_tag*_i32"
; SJLJ: call testSetjmp

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
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0
; Function Attrs: noreturn
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1
declare i8* @malloc(i32)
declare void @free(i8*)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }

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

; WASM-EH-EM-EH: LLVM ERROR: -exception-model=wasm not allowed with -enable-emscripten-cxx-exceptions
