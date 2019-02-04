; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -exception-model=wasm
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -exception-model=wasm -mattr=+exception-handling -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap %s
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-keep-registers -exception-model=wasm -mattr=+exception-handling

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.Cleanup = type { i8 }

@_ZTIi = external constant i8*

; CHECK-LABEL: test_throw:
; CHECK: throw __cpp_exception@EVENT, $0
; CHECK-NOT:  unreachable
define void @test_throw(i8* %p) {
  call void @llvm.wasm.throw(i32 0, i8* %p)
  ret void
}

; CHECK-LABEL: test_rethrow:
; CHECK:      rethrow
; CHECK-NOT:  unreachable
define void @test_rethrow(i8* %p) {
  call void @llvm.wasm.rethrow()
  ret void
}

; CHECK-LABEL: test_catch_rethrow:
; CHECK:   global.get  ${{.+}}=, __stack_pointer@GLOBAL
; CHECK:   try
; CHECK:   call      foo@FUNCTION
; CHECK:   catch     $[[EXCEPT_REF:[0-9]+]]=
; CHECK:   global.set  __stack_pointer@GLOBAL
; CHECK:   block i32
; CHECK:   br_on_exn 0, __cpp_exception@EVENT, $[[EXCEPT_REF]]
; CHECK:   rethrow
; CHECK:   end_block
; CHECK:   extract_exception $[[EXN:[0-9]+]]=
; CHECK-DAG:   i32.store  __wasm_lpad_context
; CHECK-DAG:   i32.store  __wasm_lpad_context+4
; CHECK:   i32.call  $drop=, _Unwind_CallPersonality@FUNCTION, $[[EXN]]
; CHECK:   i32.call  $drop=, __cxa_begin_catch@FUNCTION
; CHECK:   call      __cxa_end_catch@FUNCTION
; CHECK:   call      __cxa_rethrow@FUNCTION
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
; CHECK:   catch
; CHECK:   global.set  __stack_pointer@GLOBAL
; CHECK:   i32.call  $drop=, _ZN7CleanupD1Ev@FUNCTION
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

; CHECK-LABEL: test_terminatepad
; CHECK:   catch
; CHECK:   block     i32
; CHECK:   br_on_exn   0, __cpp_exception@EVENT
; CHECK:   call      __clang_call_terminate@FUNCTION, 0
; CHECK:   unreachable
; CHECK:   end_block
; CHECK:   extract_exception
; CHECK:   call      __clang_call_terminate@FUNCTION
; CHECK:   unreachable
define void @test_terminatepad() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  invoke void @foo() [ "funclet"(token %1) ]
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %catch.start
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont1
  ret void

ehcleanup:                                        ; preds = %catch.start
  %5 = cleanuppad within %1 []
  invoke void @__cxa_end_catch() [ "funclet"(token %5) ]
          to label %invoke.cont2 unwind label %terminate

invoke.cont2:                                     ; preds = %ehcleanup
  cleanupret from %5 unwind to caller

terminate:                                        ; preds = %ehcleanup
  %6 = cleanuppad within %5 []
  %7 = call i8* @llvm.wasm.get.exception(token %6)
  call void @__clang_call_terminate(i8* %7) [ "funclet"(token %6) ]
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
; CHECK:  catch
; CHECK-NOT:  global.get  $push{{.+}}=, __stack_pointer@GLOBAL
; CHECK:  global.set  __stack_pointer@GLOBAL
; CHECK:  try
; CHECK:  call      foo@FUNCTION
; CHECK:  catch
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
declare void @llvm.wasm.throw(i32, i8*)
declare void @llvm.wasm.rethrow()
declare i8* @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__cxa_rethrow()
declare void @__clang_call_terminate(i8*)
declare %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* returned)

; CHECK: __cpp_exception:
; CHECK: .eventtype  __cpp_exception i32
