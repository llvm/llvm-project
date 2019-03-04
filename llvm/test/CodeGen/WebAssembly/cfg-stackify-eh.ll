; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@_ZTIi = external constant i8*
@_ZTId = external constant i8*

; Simple test case with two catch clauses
;
; void foo();
; void test0() {
;   try {
;     foo();
;   } catch (int) {
;   } catch (double) {
;   }
; }

; CHECK-LABEL: test0
; CHECK: try
; CHECK:   call      foo
; CHECK: catch
; CHECK:   block
; CHECK:     br_if     0, {{.*}}                       # 0: down to label2
; CHECK:     i32.call  $drop=, __cxa_begin_catch
; CHECK:     call      __cxa_end_catch
; CHECK:     br        1                               # 1: down to label0
; CHECK:   end_block                                   # label2:
; CHECK:   block
; CHECK:     br_if     0, {{.*}}                       # 0: down to label3
; CHECK:     i32.call  $drop=, __cxa_begin_catch
; CHECK:     call      __cxa_end_catch
; CHECK:     br        1                               # 1: down to label0
; CHECK:   end_block                                   # label3:
; CHECK:   call      __cxa_rethrow
; CHECK: end_try                                       # label0:
define void @test0() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*), i8* bitcast (i8** @_ZTId to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch2, label %catch.fallthrough

catch2:                                           ; preds = %catch.start
  %5 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

catch.fallthrough:                                ; preds = %catch.start
  %6 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTId to i8*))
  %matches1 = icmp eq i32 %3, %6
  br i1 %matches1, label %catch, label %rethrow

catch:                                            ; preds = %catch.fallthrough
  %7 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

rethrow:                                          ; preds = %catch.fallthrough
  call void @__cxa_rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont:                                         ; preds = %entry, %catch, %catch2
  ret void
}

; Nested try-catches within a catch
; void test1() {
;   try {
;     foo();
;   } catch (int) {
;     try {
;       foo();
;     } catch (int) {
;       foo();
;     }
;   }
; }

; CHECK-LABEL: test1
; CHECK: try
; CHECK:   call      foo
; CHECK: catch
; CHECK:   block
; CHECK:     block
; CHECK:       br_if     0, {{.*}}                     # 0: down to label7
; CHECK:       i32.call  $drop=, __cxa_begin_catch
; CHECK:       try
; CHECK:         call      foo
; CHECK:         br        2                           # 2: down to label6
; CHECK:       catch
; CHECK:         try
; CHECK:           block
; CHECK:             br_if     0, {{.*}}               # 0: down to label11
; CHECK:             i32.call  $drop=, __cxa_begin_catch
; CHECK:             try
; CHECK:               call      foo
; CHECK:               br        2                     # 2: down to label9
; CHECK:             catch
; CHECK:               call      __cxa_end_catch
; CHECK:               rethrow                         # down to catch3
; CHECK:             end_try
; CHECK:           end_block                           # label11:
; CHECK:           call      __cxa_rethrow
; CHECK:           unreachable
; CHECK:         catch     {{.*}}                      # catch3:
; CHECK:           call      __cxa_end_catch
; CHECK:           rethrow                             # to caller
; CHECK:         end_try                               # label9:
; CHECK:         call      __cxa_end_catch
; CHECK:         br        2                           # 2: down to label6
; CHECK:       end_try
; CHECK:     end_block                                 # label7:
; CHECK:     call      __cxa_rethrow
; CHECK:     unreachable
; CHECK:   end_block                                   # label6:
; CHECK:   call      __cxa_end_catch
; CHECK: end_try
define void @test1() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %try.cont11 unwind label %catch.dispatch

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
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  invoke void @foo() [ "funclet"(token %1) ]
          to label %try.cont unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %catch
  %8 = catchswitch within %1 [label %catch.start3] unwind label %ehcleanup9

catch.start3:                                     ; preds = %catch.dispatch2
  %9 = catchpad within %8 [i8* bitcast (i8** @_ZTIi to i8*)]
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  %11 = call i32 @llvm.wasm.get.ehselector(token %9)
  %12 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches4 = icmp eq i32 %11, %12
  br i1 %matches4, label %catch6, label %rethrow5

catch6:                                           ; preds = %catch.start3
  %13 = call i8* @__cxa_begin_catch(i8* %10) [ "funclet"(token %9) ]
  %14 = bitcast i8* %13 to i32*
  %15 = load i32, i32* %14, align 4
  invoke void @foo() [ "funclet"(token %9) ]
          to label %invoke.cont8 unwind label %ehcleanup

invoke.cont8:                                     ; preds = %catch6
  call void @__cxa_end_catch() [ "funclet"(token %9) ]
  catchret from %9 to label %try.cont

rethrow5:                                         ; preds = %catch.start3
  invoke void @__cxa_rethrow() [ "funclet"(token %9) ]
          to label %unreachable unwind label %ehcleanup9

try.cont:                                         ; preds = %catch, %invoke.cont8
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont11

rethrow:                                          ; preds = %catch.start
  call void @__cxa_rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont11:                                       ; preds = %entry, %try.cont
  ret void

ehcleanup:                                        ; preds = %catch6
  %16 = cleanuppad within %9 []
  call void @__cxa_end_catch() [ "funclet"(token %16) ]
  cleanupret from %16 unwind label %ehcleanup9

ehcleanup9:                                       ; preds = %ehcleanup, %rethrow5, %catch.dispatch2
  %17 = cleanuppad within %1 []
  call void @__cxa_end_catch() [ "funclet"(token %17) ]
  cleanupret from %17 unwind to caller

unreachable:                                      ; preds = %rethrow5
  unreachable
}

; Nested loop within a catch clause
; void test2() {
;   try {
;     foo();
;   } catch (...) {
;     for (int i = 0; i < 50; i++)
;       foo();
;   }
; }

; CHECK-LABEL: test2
; CHECK: try
; CHECK:   call      foo
; CHECK: catch
; CHECK:   i32.call  $drop=, __cxa_begin_catch
; CHECK:   loop                                        # label15:
; CHECK:     block
; CHECK:       block
; CHECK:         br_if     0, {{.*}}                   # 0: down to label17
; CHECK:         try
; CHECK:           call      foo
; CHECK:           br        2                         # 2: down to label16
; CHECK:         catch
; CHECK:           try
; CHECK:             call      __cxa_end_catch
; CHECK:           catch
; CHECK:             call      __clang_call_terminate
; CHECK:             unreachable
; CHECK:           end_try
; CHECK:           rethrow                             # to caller
; CHECK:         end_try
; CHECK:       end_block                               # label17:
; CHECK:       call      __cxa_end_catch
; CHECK:       br        2                             # 2: down to label13
; CHECK:     end_block                                 # label16:
; CHECK:     br        0                               # 0: up to label15
; CHECK:   end_loop
; CHECK: end_try                                       # label13:
define void @test2() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
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
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %catch.start
  %i.0 = phi i32 [ 0, %catch.start ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 50
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  invoke void @foo() [ "funclet"(token %1) ]
          to label %for.inc unwind label %ehcleanup

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %for.end, %entry
  ret void

ehcleanup:                                        ; preds = %for.body
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

declare void @foo()
declare void @bar()
declare i32 @__gxx_wasm_personality_v0(...)
declare i8* @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__cxa_rethrow()
declare void @__clang_call_terminate(i8*)
declare void @_ZSt9terminatev()
