; RUN: opt < %s -wasm-lower-em-ehsjlj -emscripten-cxx-exceptions-whitelist=do_catch -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define void @dont_catch() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @dont_catch(
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %lpad
; CHECK: entry:
; CHECK-NEXT: call void @foo()
; CHECK-NEXT: br label %invoke.cont

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  br label %catch

catch:                                            ; preds = %lpad
  %3 = call i8* @__cxa_begin_catch(i8* %1)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void
}

define void @do_catch() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @do_catch(
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %lpad
; CHECK: entry:
; CHECK-NEXT: store i32 0, i32*
; CHECK-NEXT: call void @__invoke_void(void ()* @foo)

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  br label %catch

catch:                                            ; preds = %lpad
  %3 = call i8* @__cxa_begin_catch(i8* %1)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void
}

declare void @foo()
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
