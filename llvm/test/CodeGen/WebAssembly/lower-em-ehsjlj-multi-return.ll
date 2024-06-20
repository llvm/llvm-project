; RUN: not --crash llc < %s -enable-emscripten-cxx-exceptions -mattr=+multivalue -target-abi=experimental-mv 2>&1 | FileCheck %s --check-prefix=EH
; RUN: not --crash llc < %s -enable-emscripten-sjlj -mattr=+multivalue -target-abi=experimental-mv 2>&1 | FileCheck %s --check-prefix=SJLJ

; Currently multivalue returning functions are not supported in Emscripten EH /
; SjLj. Make sure they error out.

target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

define void @exception() personality ptr @__gxx_personality_v0 {
entry:
  invoke {i32, i32} @foo(i32 3)
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { ptr, i32 }
          catch ptr null
  %2 = extractvalue { ptr, i32 } %1, 0
  %3 = extractvalue { ptr, i32 } %1, 1
  %4 = call ptr @__cxa_begin_catch(ptr %2) #2
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  ret void
}

define void @setjmp_longjmp() {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  call {i32, i32} @foo(i32 3)
  call void @longjmp(ptr %buf, i32 1) #1
  unreachable
}

declare {i32, i32} @foo(i32)
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

; EH: LLVM ERROR: Emscripten EH/SjLj does not support multivalue returns
; SJLJ: LLVM ERROR: Emscripten EH/SjLj does not support multivalue returns
