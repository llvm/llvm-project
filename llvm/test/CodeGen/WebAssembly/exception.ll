; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -exception-model=wasm
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -exception-model=wasm -mattr=+exception-handling -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap %s
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-keep-registers -exception-model=wasm -mattr=+exception-handling

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.Cleanup = type { i8 }

@_ZTIi = external constant i8*

declare void @llvm.wasm.throw(i32, i8*)

; CHECK-LABEL: test_throw:
; CHECK:      local.get $push0=, 0
; CHECK-NEXT: throw __cpp_exception@EVENT, $pop0
; CHECK-NOT:  unreachable
define void @test_throw(i8* %p) {
  call void @llvm.wasm.throw(i32 0, i8* %p)
  ret void
}

; CHECK-LABEL: test_catch_rethrow:
; CHECK:   global.get  $push{{.+}}=, __stack_pointer@GLOBAL
; CHECK:   try
; CHECK:   call      foo@FUNCTION
; CHECK:   i32.catch     $push{{.+}}=, 0
; CHECK:   global.set  __stack_pointer@GLOBAL
; CHECK-DAG:   i32.store  __wasm_lpad_context
; CHECK-DAG:   i32.store  __wasm_lpad_context+4
; CHECK:   i32.call  $push{{.+}}=, _Unwind_CallPersonality@FUNCTION
; CHECK:   i32.call  $push{{.+}}=, __cxa_begin_catch@FUNCTION
; CHECK:   call      __cxa_end_catch@FUNCTION
; CHECK:   call      __cxa_rethrow@FUNCTION
; CHECK-NEXT:   rethrow
; CHECK:   end_try
define void @test_catch_rethrow() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
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
; CHECK:   try
; CHECK:   call      foo@FUNCTION
; CHECK:   catch_all
; CHECK:   global.set  __stack_pointer@GLOBAL
; CHECK:   i32.call  $push{{.+}}=, _ZN7CleanupD1Ev@FUNCTION
; CHECK:   rethrow
; CHECK:   end_try
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

; - Tests multple terminate pads are merged into one
; - Tests a catch_all terminate pad is created after a catch terminate pad

; CHECK-LABEL: test_terminatepad
; CHECK:  i32.catch
; CHECK:  call      __clang_call_terminate@FUNCTION
; CHECK:  unreachable
; CHECK:  catch_all
; CHECK:  call      _ZSt9terminatev@FUNCTION
; CHECK-NOT:  call      __clang_call_terminate@FUNCTION
define hidden i32 @test_terminatepad() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %c = alloca %struct.Cleanup, align 1
  %c1 = alloca %struct.Cleanup, align 1
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %call = invoke %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %c1)
          to label %try.cont unwind label %catch.dispatch

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  %call4 = invoke %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %c1) [ "funclet"(token %0) ]
          to label %invoke.cont3 unwind label %terminate

invoke.cont3:                                     ; preds = %ehcleanup
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont3, %invoke.cont
  %1 = catchswitch within none [label %catch.start] unwind label %ehcleanup7

catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* null]
  %3 = call i8* @llvm.wasm.get.exception(token %2)
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  %5 = call i8* @__cxa_begin_catch(i8* %3) [ "funclet"(token %2) ]
  invoke void @__cxa_end_catch() [ "funclet"(token %2) ]
          to label %invoke.cont5 unwind label %ehcleanup7

invoke.cont5:                                     ; preds = %catch.start
  catchret from %2 to label %try.cont

try.cont:                                         ; preds = %invoke.cont5, %invoke.cont
  %call6 = call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %c)
  ret i32 0

ehcleanup7:                                       ; preds = %catch.start, %catch.dispatch
  %6 = cleanuppad within none []
  %call9 = invoke %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %c) [ "funclet"(token %6) ]
          to label %invoke.cont8 unwind label %terminate10

invoke.cont8:                                     ; preds = %ehcleanup7
  cleanupret from %6 unwind to caller

terminate:                                        ; preds = %ehcleanup
  %7 = cleanuppad within %0 []
  %8 = call i8* @llvm.wasm.get.exception(token %7)
  call void @__clang_call_terminate(i8* %8) [ "funclet"(token %7) ]
  unreachable

terminate10:                                      ; preds = %ehcleanup7
  %9 = cleanuppad within %6 []
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  call void @__clang_call_terminate(i8* %10) [ "funclet"(token %9) ]
  unreachable
}

; Tests prologues and epilogues are not generated within EH scopes.
; They should not be treated as funclets; BBs starting with a catch instruction
; should not have a prologue, and BBs ending with a catchret/cleanupret should
; not have an epilogue. This is separate from __stack_pointer restoring
; instructions after a catch instruction.

; CHECK-LABEL: test_no_prolog_epilog_in_ehpad
; CHECK:  try
; CHECK:  call      foo@FUNCTION
; CHECK:  i32.catch
; CHECK-NOT:  global.get  $push{{.+}}=, __stack_pointer@GLOBAL
; CHECK:  global.set  __stack_pointer@GLOBAL
; CHECK:  try
; CHECK:  call      foo@FUNCTION
; CHECK:  catch_all
; CHECK-NOT:  global.get  $push{{.+}}=, __stack_pointer@GLOBAL
; CHECK:  global.set  __stack_pointer@GLOBAL
; CHECK:  call      __cxa_end_catch@FUNCTION
; CHECK-NOT:  global.set  __stack_pointer@GLOBAL, $pop{{.+}}
; CHECK:  end_try
; CHECK-NOT:  global.set  __stack_pointer@GLOBAL, $pop{{.+}}
; CHECK:  end_try
define void @test_no_prolog_epilog_in_ehpad() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %stack_var = alloca i32, align 4
  call void @bar(i32* %stack_var)
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
  %6 = bitcast i8* %5 to float*
  %7 = load float, float* %6, align 4
  invoke void @foo() [ "funclet"(token %1) ]
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %catch
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

rethrow:                                          ; preds = %catch.start
  call void @__cxa_rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont:                                         ; preds = %entry, %invoke.cont1
  ret void

ehcleanup:                                        ; preds = %catch
  %8 = cleanuppad within %1 []
  call void @__cxa_end_catch() [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; When a function does not have stack-allocated objects, it does not need to
; store SP back to __stack_pointer global at the epilog.

; CHECK-LABEL: no_sp_writeback
; CHECK:  try
; CHECK:  call foo@FUNCTION
; CHECK:  end_try
; CHECK-NOT:  global.set  __stack_pointer@GLOBAL
; CHECK:  return
define void @no_sp_writeback() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) #2 [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch.start
  ret void
}

declare void @foo()
declare void @bar(i32*)
declare i32 @__gxx_wasm_personality_v0(...)
declare i8* @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__cxa_rethrow()
declare void @__clang_call_terminate(i8*)
declare void @_ZSt9terminatev()
declare %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* returned)

; CHECK: __cpp_exception:
; CHECK: .eventtype  __cpp_exception i32
