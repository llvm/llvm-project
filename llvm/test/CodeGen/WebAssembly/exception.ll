; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -exception-model=wasm | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.Cleanup = type { i8 }

@_ZTIi = external constant i8*

declare void @llvm.wasm.throw(i32, i8*)

; CHECK-LABEL: test_throw:
; CHECK-NEXT: i32.const $push0=, 0
; CHECK-NEXT: throw 0, $pop0
define void @test_throw() {
  call void @llvm.wasm.throw(i32 0, i8* null)
  ret void
}

; CHECK-LABEL: test_catch:
; CHECK:   call      foo@FUNCTION
; CHECK:   i32.catch     $push{{.+}}=, 0
; CHECK-DAG:   i32.store  __wasm_lpad_context
; CHECK-DAG:   i32.store  __wasm_lpad_context+4
; CHECK:   i32.call  $push{{.+}}=, _Unwind_CallPersonality@FUNCTION
; CHECK:   i32.call  $push{{.+}}=, __cxa_begin_catch@FUNCTION
; CHECK:   call      __cxa_end_catch@FUNCTION
; CHECK:   call      __cxa_rethrow@FUNCTION
define void @test_catch() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  %5 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

rethrow:                                          ; preds = %catch.start
  call void @__cxa_rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont:                                         ; preds = %entry, %catch
  ret void
}

; CHECK-LABEL: test_cleanup:
; CHECK:   call      foo@FUNCTION
; CHECK:   return
; CHECK:   i32.call  $push20=, _ZN7CleanupD1Ev@FUNCTION
; CHECK:   rethrow   0
define void @test_cleanup() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %c = alloca %struct.Cleanup, align 1
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %call = call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %c)
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  %call1 = call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %c) [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

declare void @foo()
declare void @func(i32)
declare i32 @__gxx_wasm_personality_v0(...)
declare i8* @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__cxa_rethrow()
declare void @__clang_call_terminate(i8*)
declare %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* returned)
