; REQUIRES: asserts
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling | FileCheck %s
; RUN: llc < %s -O0 -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -verify-machineinstrs -exception-model=wasm -mattr=+exception-handling | FileCheck %s --check-prefix=NOOPT
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling -wasm-disable-ehpad-sort | FileCheck %s --check-prefix=NOSORT
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling -wasm-disable-ehpad-sort -stats 2>&1 | FileCheck %s --check-prefix=NOSORT-STAT

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
; CHECK:   rethrow   {{.*}}                            # to caller
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
  call void @llvm.wasm.rethrow.in.catch() [ "funclet"(token %1) ]
  unreachable

try.cont:                                         ; preds = %catch, %catch2, %entry
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
; CHECK:               rethrow   {{.*}}                # down to catch3
; CHECK:             end_try
; CHECK:           end_block                           # label11:
; CHECK:           rethrow   {{.*}}                    # down to catch3
; CHECK:         catch     {{.*}}                      # catch3:
; CHECK:           call      __cxa_end_catch
; CHECK:           rethrow   {{.*}}                    # to caller
; CHECK:         end_try                               # label9:
; CHECK:         call      __cxa_end_catch
; CHECK:         br        2                           # 2: down to label6
; CHECK:       end_try
; CHECK:     end_block                                 # label7:
; CHECK:     rethrow   {{.*}}                          # to caller
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
  invoke void @llvm.wasm.rethrow.in.catch() [ "funclet"(token %9) ]
          to label %unreachable unwind label %ehcleanup9

try.cont:                                         ; preds = %invoke.cont8, %catch
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont11

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow.in.catch() [ "funclet"(token %1) ]
  unreachable

try.cont11:                                       ; preds = %try.cont, %entry
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
; CHECK:           rethrow   {{.*}}                    # to caller
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

; Tests if block and try markers are correctly placed. Even if two predecessors
; of the EH pad are bb2 and bb3 and their nearest common dominator is bb1, the
; TRY marker should be placed at bb0 because there's a branch from bb0 to bb2,
; and scopes cannot be interleaved.

; NOOPT-LABEL: test3
; NOOPT: try
; NOOPT:   block
; NOOPT:     block
; NOOPT:       block
; NOOPT:       end_block
; NOOPT:     end_block
; NOOPT:     call      foo
; NOOPT:   end_block
; NOOPT:   call      bar
; NOOPT: catch     {{.*}}
; NOOPT: end_try
define void @test3() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  br i1 undef, label %bb1, label %bb2

bb1:                                              ; preds = %bb0
  br i1 undef, label %bb3, label %bb4

bb2:                                              ; preds = %bb0
  br label %try.cont

bb3:                                              ; preds = %bb1
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

bb4:                                              ; preds = %bb1
  invoke void @bar()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %bb4, %bb3
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.start, %bb4, %bb3, %bb2
  ret void
}

; Tests if try/end_try markers are placed correctly wrt loop/end_loop markers,
; when try and loop markers are in the same BB and end_try and end_loop are in
; another BB.
; CHECK: loop
; CHECK:   try
; CHECK:     call      foo
; CHECK:   catch
; CHECK:   end_try
; CHECK: end_loop
define void @test4(i32* %p) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  store volatile i32 0, i32* %p
  br label %loop

loop:                                             ; preds = %try.cont, %entry
  store volatile i32 1, i32* %p
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %loop
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.start, %loop
  br label %loop
}

; Some of test cases below are hand-tweaked by deleting some library calls to
; simplify tests and changing the order of basic blocks to cause unwind
; destination mismatches. And we use -wasm-disable-ehpad-sort to create maximum
; number of mismatches in several tests below.

; 'call bar''s original unwind destination was 'catch14', but after control flow
; linearization, its unwind destination incorrectly becomes 'catch15'. We fix
; this by wrapping the call with a nested try/catch/end_try and branching to the
; right destination (label32).

; NOSORT-LABEL: test5
; NOSORT:   block
; NOSORT:     try
; NOSORT:       try
; NOSORT:         call      foo
; --- Nested try/catch/end_try starts
; NOSORT:         try
; NOSORT:           call      bar
; NOSORT:         catch     $drop=
; NOSORT:           br        2                        # 2: down to label32
; NOSORT:         end_try
; --- Nested try/catch/end_try ends
; NOSORT:         br        2                          # 2: down to label31
; NOSORT:       catch     $drop=                       # catch15:
; NOSORT:         br        2                          # 2: down to label31
; NOSORT:       end_try
; NOSORT:     catch     $drop=                         # catch14:
; NOSORT:     end_try                                  # label32:
; NOSORT:   end_block                                  # label31:
; NOSORT:   return

define void @test5() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %catch.dispatch0

bb1:                                              ; preds = %bb0
  invoke void @bar()
          to label %try.cont unwind label %catch.dispatch1

catch.dispatch0:                                  ; preds = %bb0
  %0 = catchswitch within none [label %catch.start0] unwind to caller

catch.start0:                                     ; preds = %catch.dispatch0
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  catchret from %1 to label %try.cont

catch.dispatch1:                                  ; preds = %bb1
  %4 = catchswitch within none [label %catch.start1] unwind to caller

catch.start1:                                     ; preds = %catch.dispatch1
  %5 = catchpad within %4 [i8* null]
  %6 = call i8* @llvm.wasm.get.exception(token %5)
  %7 = call i32 @llvm.wasm.get.ehselector(token %5)
  catchret from %5 to label %try.cont

try.cont:                                         ; preds = %catch.start1, %catch.start0, %bb1
  ret void
}

; Two 'call bar''s original unwind destination was the caller, but after control
; flow linearization, their unwind destination incorrectly becomes 'catch17'. We
; fix this by wrapping the call with a nested try/catch/end_try and branching to
; the right destination (label4), from which we rethrow the exception to the
; caller.

; And the return value of 'baz' should NOT be stackified because the BB is split
; during fixing unwind mismatches.

; NOSORT-LABEL: test6
; NOSORT:   try
; NOSORT:     call      foo
; --- Nested try/catch/end_try starts
; NOSORT:     try
; NOSORT:       call      bar
; NOSORT:       i32.call  ${{[0-9]+}}=, baz
; NOSORT-NOT:   i32.call  $push{{.*}}=, baz
; NOSORT:     catch     $[[REG:[0-9]+]]=
; NOSORT:       br        1                            # 1: down to label35
; NOSORT:     end_try
; --- Nested try/catch/end_try ends
; NOSORT:     return
; NOSORT:   catch     $drop=                           # catch17:
; NOSORT:     return
; NOSORT:   end_try                                    # label35:
; NOSORT:   rethrow   $[[REG]]                         # to caller

define void @test6() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %catch.dispatch0

bb1:                                              ; preds = %bb0
  call void @bar()
  %call = call i32 @baz()
  call void @nothrow(i32 %call) #0
  ret void

catch.dispatch0:                                  ; preds = %bb0
  %0 = catchswitch within none [label %catch.start0] unwind to caller

catch.start0:                                     ; preds = %catch.dispatch0
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.start0
  ret void
}

; If not for the unwind destination mismatch, the LOOP marker here would have an
; i32 signature. But because we add a rethrow instruction at the end of the
; appendix block, now the LOOP marker does not have a signature (= has a void
; signature). Here the two calls two 'bar' are supposed to throw up to the
; caller, but incorrectly unwind to 'catch19' after linearizing the CFG.

; NOSORT-LABEL: test7
; NOSORT: block
; NOSORT-NOT: loop      i32
; NOSORT:   loop                                       # label38:
; NOSORT:     try
; NOSORT:       call      foo
; --- Nested try/catch/end_try starts
; NOSORT:       try
; NOSORT:         call      bar
; NOSORT:         call      bar
; NOSORT:       catch     $[[REG:[0-9]+]]=
; NOSORT:         br        1                          # 1: down to label39
; NOSORT:       end_try
; --- Nested try/catch/end_try ends
; NOSORT:       return    {{.*}}
; NOSORT:     catch     $drop=                         # catch19:
; NOSORT:       br        1                            # 1: up to label38
; NOSORT:     end_try                                  # label39:
; NOSORT:   end_loop
; NOSORT: end_block
; NOSORT: rethrow   $[[REG]]                           # to caller

define i32 @test7(i32* %p) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  store volatile i32 0, i32* %p
  br label %loop

loop:                                             ; preds = %try.cont, %entry
  store volatile i32 1, i32* %p
  invoke void @foo()
          to label %bb unwind label %catch.dispatch

bb:                                               ; preds = %loop
  call void @bar()
  call void @bar()
  ret i32 0

catch.dispatch:                                   ; preds = %loop
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.start
  br label %loop
}

; When we have both kinds of EH pad unwind mismatches:
; - A may-throw instruction unwinds to an incorrect EH pad after linearizing the
;   CFG, when it is supposed to unwind to another EH pad.
; - A may-throw instruction unwinds to an incorrect EH pad after linearizing the
;   CFG, when it is supposed to unwind to the caller.

; NOSORT-LABEL: test8
; NOSORT: block
; NOSORT:   block
; NOSORT:     try
; NOSORT:       try
; NOSORT:         call      foo
; --- Nested try/catch/end_try starts
; NOSORT:         try
; NOSORT:           call      bar
; NOSORT:         catch     $[[REG0:[0-9]+]]=
; NOSORT:           br        2                        # 2: down to label43
; NOSORT:         end_try
; --- Nested try/catch/end_try ends
; NOSORT:         br        2                          # 2: down to label42
; NOSORT:       catch     {{.*}}
; NOSORT:         block     i32
; NOSORT:           br_on_exn   0, {{.*}}              # 0: down to label46
; --- Nested try/catch/end_try starts
; NOSORT:           try
; NOSORT:             rethrow   {{.*}}                 # down to catch24
; NOSORT:           catch     $[[REG1:[0-9]+]]=        # catch24:
; NOSORT:             br        5                      # 5: down to label41
; NOSORT:           end_try
; --- Nested try/catch/end_try ends
; NOSORT:         end_block                            # label46:
; NOSORT:         i32.call  $drop=, __cxa_begin_catch
; --- Nested try/catch/end_try starts
; NOSORT:         try
; NOSORT:           call      __cxa_end_catch
; NOSORT:         catch     $[[REG1]]=
; NOSORT:           br        4                        # 4: down to label41
; NOSORT:         end_try
; --- Nested try/catch/end_try ends
; NOSORT:         br        2                          # 2: down to label42
; NOSORT:       end_try
; NOSORT:     catch     $[[REG0]]=
; NOSORT:     end_try                                  # label43:
; NOSORT:     i32.call  $drop=, __cxa_begin_catch
; NOSORT:     call      __cxa_end_catch
; NOSORT:   end_block                                  # label42:
; NOSORT:   return
; NOSORT: end_block                                    # label41:
; NOSORT: rethrow   $[[REG1]]                          # to caller
define void @test8() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %catch.dispatch0

bb1:                                              ; preds = %bb0
  invoke void @bar()
          to label %try.cont unwind label %catch.dispatch1

catch.dispatch0:                                  ; preds = %bb0
  %0 = catchswitch within none [label %catch.start0] unwind to caller

catch.start0:                                     ; preds = %catch.dispatch0
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

catch.dispatch1:                                  ; preds = %bb1
  %5 = catchswitch within none [label %catch.start1] unwind to caller

catch.start1:                                     ; preds = %catch.dispatch1
  %6 = catchpad within %5 [i8* null]
  %7 = call i8* @llvm.wasm.get.exception(token %6)
  %8 = call i32 @llvm.wasm.get.ehselector(token %6)
  %9 = call i8* @__cxa_begin_catch(i8* %7) [ "funclet"(token %6) ]
  call void @__cxa_end_catch() [ "funclet"(token %6) ]
  catchret from %6 to label %try.cont

try.cont:                                         ; preds = %catch.start1, %catch.start0, %bb1
  ret void
}

; In CFGSort, EH pads should be sorted as soon as it is available and
; 'Preferred' queue and should NOT be entered into 'Ready' queue unless we are
; in the middle of sorting another region that does not contain the EH pad. In
; this example, 'catch.start' should be sorted right after 'if.then' is sorted
; (before 'cont' is sorted) and there should not be any unwind destination
; mismatches in CFGStackify.

; NOOPT: block
; NOOPT:   try
; NOOPT:     call      foo
; NOOPT:   catch
; NOOPT:   end_try
; NOOPT:   call      foo
; NOOPT: end_block
; NOOPT: return
define void @test9(i32 %arg) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %tobool = icmp ne i32 %arg, 0
  br i1 %tobool, label %if.then, label %if.end

catch.dispatch:                                   ; preds = %if.then
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %if.end

if.then:                                          ; preds = %entry
  invoke void @foo()
          to label %cont unwind label %catch.dispatch

cont:                                             ; preds = %if.then
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %cont, %catch.start, %entry
  ret void
}

%class.Object = type { i8 }

; Intrinsics like memcpy, memmove, and memset don't throw and are lowered into
; calls to external symbols (not global addresses) in instruction selection,
; which will be eventually lowered to library function calls.
; Because this test runs with -wasm-disable-ehpad-sort, these library calls in
; invoke.cont BB fall within try~end_try, but they shouldn't cause crashes or
; unwinding destination mismatches in CFGStackify.

; NOSORT-LABEL: test10
; NOSORT: try
; NOSORT:   call  foo
; NOSORT:   i32.call {{.*}} memcpy
; NOSORT:   i32.call {{.*}} memmove
; NOSORT:   i32.call {{.*}} memset
; NOSORT:   return
; NOSORT: catch
; NOSORT:   rethrow
; NOSORT: end_try
define void @test10(i8* %a, i8* %b) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %o = alloca %class.Object, align 1
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 100, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %a, i8* %b, i32 100, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %a, i8 0, i32 100, i1 false)
  %call = call %class.Object* @_ZN6ObjectD2Ev(%class.Object* %o) #1
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  %call2 = call %class.Object* @_ZN6ObjectD2Ev(%class.Object* %o) #1 [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

; Tests if 'try' marker is placed correctly. In this test, 'try' should be
; placed before the call to 'nothrow_i32' and not between the call to
; 'nothrow_i32' and 'fun', because the return value of 'nothrow_i32' is
; stackified and pushed onto the stack to be consumed by the call to 'fun'.

; CHECK-LABEL: test11
; CHECK: try
; CHECK: i32.call  $push{{.*}}=, nothrow_i32
; CHECK: call      fun, $pop{{.*}}
define void @test11() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %call = call i32 @nothrow_i32()
  invoke void @fun(i32 %call)
          to label %invoke.cont unwind label %terminate

invoke.cont:                                      ; preds = %entry
  ret void

terminate:                                        ; preds = %entry
  %0 = cleanuppad within none []
  %1 = tail call i8* @llvm.wasm.get.exception(token %0)
  call void @__clang_call_terminate(i8* %1) [ "funclet"(token %0) ]
  unreachable
}

; Check if the unwind destination mismatch stats are correct
; NOSORT-STAT: 11 wasm-cfg-stackify    - Number of EH pad unwind mismatches found

declare void @foo()
declare void @bar()
declare i32 @baz()
declare void @fun(i32)
; Function Attrs: nounwind
declare void @nothrow(i32) #0
declare i32 @nothrow_i32() #0
; Function Attrs: nounwind
declare %class.Object* @_ZN6ObjectD2Ev(%class.Object* returned) #0
declare i32 @__gxx_wasm_personality_v0(...)
declare i8* @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare void @llvm.wasm.rethrow.in.catch()
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__clang_call_terminate(i8*)
declare void @_ZSt9terminatev()
; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg) #0
; Function Attrs: nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1 immarg) #0
; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1 immarg) #0

attributes #0 = { nounwind }
