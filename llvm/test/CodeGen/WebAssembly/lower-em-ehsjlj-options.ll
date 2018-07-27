; RUN: llc < %s -enable-emscripten-cxx-exceptions | FileCheck %s --check-prefix=EH
; RUN: llc < %s -enable-emscripten-sjlj | FileCheck %s --check-prefix=SJLJ
; RUN: llc < %s | FileCheck %s --check-prefix=NONE

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

define hidden void @exception() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; EH-LABEL:   type exception,@function
; NONE-LABEL: type exception,@function
entry:
  invoke void @foo()
          to label %try.cont unwind label %lpad
; EH:   call __invoke_void@FUNCTION
; NONE: call foo@FUNCTION

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

define hidden void @setjmp_longjmp() {
; SJLJ-LABEL: type setjmp_longjmp,@function
; NONE-LABEL: type setjmp_longjmp,@function
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable
; SJLJ: i32.call saveSetjmp@FUNCTION
; SJLJ: i32.call testSetjmp@FUNCTION
; NONE: i32.call setjmp@FUNCTION
; NONE: call longjmp@FUNCTION
}

declare void @foo()
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
