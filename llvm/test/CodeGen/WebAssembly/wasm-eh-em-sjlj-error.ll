; RUN: not --crash llc < %s -wasm-enable-eh -enable-emscripten-sjlj -exception-model=wasm 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

define void @wasm_eh_emscripten_sjlj_error() personality ptr @__gxx_wasm_personality_v0 {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  br label %bb

bb:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call ptr @__cxa_begin_catch(ptr %2) #2 [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch.start
  ret void
}

declare void @foo()
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nounwind
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #0
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

; CHECK: LLVM ERROR: In function wasm_eh_emscripten_sjlj_error: When using Wasm EH with Emscripten SjLj, there is a restriction that `setjmp` function call and exception cannot be used within the same function
; CHECK-NEXT: invoke void @foo()
; CHECK-NEXT:         to label %try.cont unwind label %catch.dispatch
