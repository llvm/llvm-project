; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @test(ptr %p) sanitize_address personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
; CHECK: catch.start:
; CHECK: %[[CATCHPAD0:.*]] = catchpad
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call ptr @__cxa_begin_catch(ptr %2) #2 [ "funclet"(token %1) ]
  %5 = load i32, ptr %p, align 4
; This __asan_report_load4 is genereated within a newly created BB, but it
; has the correct "funclet" op bundle.
; CHECK: {{.*}}:
; CHECK: call void @__asan_report_load4(i32 %{{.*}}) {{.*}} [ "funclet"(token %[[CATCHPAD0]]) ]
  invoke void @foo() [ "funclet"(token %1) ]
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %catch.start
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont1
  ret void

ehcleanup:                                        ; preds = %catch.start
  %6 = cleanuppad within %1 []
; CHECK: ehcleanup:
; CHECK: %[[CLEANUPPAD0:.*]] = cleanuppad
  store i32 42, ptr %p, align 4
; This __asan_report_store4 is genereated within a newly created BB, but it
; has the correct "funclet" op bundle.
; CHECK: {{.*}}:
; CHECK: call void @__asan_report_store4(i32 %{{.*}}) {{.*}} [ "funclet"(token %[[CLEANUPPAD0]]) ]
  invoke void @__cxa_end_catch() [ "funclet"(token %6) ]
          to label %invoke.cont2 unwind label %terminate

invoke.cont2:                                     ; preds = %ehcleanup
  cleanupret from %6 unwind to caller

terminate:                                        ; preds = %ehcleanup
  %7 = cleanuppad within %6 []
; CHECK: terminate:
; CHECK: %[[CLEANUPPAD1:.*]] = cleanuppad
  call void @_ZSt9terminatev() #3 [ "funclet"(token %7) ]
; CHECK: call void @__asan_handle_no_return() [ "funclet"(token %[[CLEANUPPAD1]]) ]
  unreachable
}

declare void @foo()
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.wasm.get.ehselector(token) #0
; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for(ptr) #1
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @_ZSt9terminatev()

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nounwind memory(none) }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind }
