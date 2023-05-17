; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions -emscripten-cxx-exceptions-allowed=do_catch -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @dont_catch() personality ptr @__gxx_personality_v0 {
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
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %catch

catch:                                            ; preds = %lpad
  %3 = call ptr @__cxa_begin_catch(ptr %1)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void
}

define void @do_catch() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @do_catch(
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %lpad
; CHECK: entry:
; CHECK-NEXT: store i32 0, ptr
; CHECK-NEXT: call cc{{.*}} void @__invoke_void(ptr @foo)

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %catch

catch:                                            ; preds = %lpad
  %3 = call ptr @__cxa_begin_catch(ptr %1)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void
}

declare void @foo()
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
