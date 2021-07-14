; RUN: not --crash llc < %s -enable-emscripten-cxx-exceptions -mattr=+multivalue 2>&1 | FileCheck %s --check-prefix=EH
; RUN: not --crash llc < %s -enable-emscripten-sjlj -mattr=+multivalue 2>&1 | FileCheck %s --check-prefix=SJLJ

; Currently multivalue returning functions are not supported in Emscripten EH /
; SjLj. Make sure they error out.

target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

define void @exception() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke {i32, i32} @foo(i32 3)
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = extractvalue { i8*, i32 } %1, 1
  %4 = call i8* @__cxa_begin_catch(i8* %2) #2
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  ret void
}

define void @setjmp_longjmp() {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call {i32, i32} @foo(i32 3)
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable
}

declare {i32, i32} @foo(i32)
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

; EH: LLVM ERROR: Emscripten EH/SjLj does not support multivalue returns
; SJLJ: LLVM ERROR: Emscripten EH/SjLj does not support multivalue returns
