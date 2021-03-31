; REQUIRES: asserts
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling | FileCheck %s
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling
; RUN: llc < %s -O0 -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -verify-machineinstrs -exception-model=wasm -mattr=+exception-handling | FileCheck %s --check-prefix=NOOPT
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling -wasm-disable-ehpad-sort -stats 2>&1 | FileCheck %s --check-prefix=NOSORT
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-block-placement -verify-machineinstrs -fast-isel=false -machine-sink-split-probability-threshold=0 -cgp-freq-ratio-to-skip-merge=1000 -exception-model=wasm -mattr=+exception-handling -wasm-disable-ehpad-sort | FileCheck %s --check-prefix=NOSORT-LOCALS

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@_ZTIi = external constant i8*
@_ZTId = external constant i8*

%class.Object = type { i8 }
%class.MyClass = type { i32 }

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
; CHECK:     br_if     0, {{.*}}                       # 0: down to label[[L0:[0-9]+]]
; CHECK:     call      $drop=, __cxa_begin_catch
; CHECK:     call      __cxa_end_catch
; CHECK:     br        1                               # 1: down to label[[L1:[0-9]+]]
; CHECK:   end_block                                   # label[[L0]]:
; CHECK:   block
; CHECK:     br_if     0, {{.*}}                       # 0: down to label[[L2:[0-9]+]]
; CHECK:     call      $drop=, __cxa_begin_catch
; CHECK:     call      __cxa_end_catch
; CHECK:     br        1                               # 1: down to label[[L1]]
; CHECK:   end_block                                   # label[[L2]]:
; CHECK:   rethrow   0                                 # to caller
; CHECK: end_try                                       # label[[L1]]:
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
  call void @llvm.wasm.rethrow() [ "funclet"(token %1) ]
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
; CHECK:   call  foo
; CHECK: catch
; CHECK:   block
; CHECK:     block
; CHECK:       br_if     0, {{.*}}                     # 0: down to label[[L0:[0-9+]]]
; CHECK:       call  $drop=, __cxa_begin_catch, $0
; CHECK:       try
; CHECK:         try
; CHECK:           call  foo
; CHECK:           br        3                         # 3: down to label[[L1:[0-9+]]]
; CHECK:         catch
; CHECK:           block
; CHECK:             block
; CHECK:               br_if     0, {{.*}}             # 0: down to label[[L2:[0-9+]]]
; CHECK:               call  $drop=, __cxa_begin_catch
; CHECK:               try
; CHECK:                 call  foo
; CHECK:                 br        2                   # 2: down to label[[L3:[0-9+]]]
; CHECK:               catch_all
; CHECK:                 call  __cxa_end_catch
; CHECK:                 rethrow   0                   # down to catch[[L4:[0-9+]]]
; CHECK:               end_try
; CHECK:             end_block                         # label[[L2]]:
; CHECK:             rethrow   1                       # down to catch[[L4]]
; CHECK:           end_block                           # label[[L3]]:
; CHECK:           call  __cxa_end_catch
; CHECK:           br        3                         # 3: down to label[[L1]]
; CHECK:         end_try
; CHECK:       catch_all                               # catch[[L4]]:
; CHECK:         call  __cxa_end_catch
; CHECK:         rethrow   0                           # to caller
; CHECK:       end_try
; CHECK:     end_block                                 # label[[L0]]:
; CHECK:     rethrow   1                               # to caller
; CHECK:   end_block                                   # label[[L1]]:
; CHECK:   call  __cxa_end_catch
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
  invoke void @llvm.wasm.rethrow() [ "funclet"(token %9) ]
          to label %unreachable unwind label %ehcleanup9

try.cont:                                         ; preds = %invoke.cont8, %catch
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont11

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow() [ "funclet"(token %1) ]
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
; CHECK:   call      $drop=, __cxa_begin_catch
; CHECK:   loop                                        # label[[L0:[0-9]+]]:
; CHECK:     block
; CHECK:       block
; CHECK:         br_if     0, {{.*}}                   # 0: down to label[[L1:[0-9]+]]
; CHECK:         try
; CHECK:           call      foo
; CHECK:           br        2                         # 2: down to label[[L2:[0-9]+]]
; CHECK:         catch
; CHECK:           try
; CHECK:             call      __cxa_end_catch
; CHECK:           catch_all
; CHECK:             call      _ZSt9terminatev
; CHECK:             unreachable
; CHECK:           end_try
; CHECK:           rethrow   0                         # to caller
; CHECK:         end_try
; CHECK:       end_block                               # label[[L1]]:
; CHECK:       call      __cxa_end_catch
; CHECK:       br        2                             # 2: down to label[[L3:[0-9]+]]
; CHECK:     end_block                                 # label[[L2]]:
; CHECK:     br        0                               # 0: up to label[[L0]]
; CHECK:   end_loop
; CHECK: end_try                                       # label[[L3]]:
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
  call void @_ZSt9terminatev() [ "funclet"(token %6) ]
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

; - Call unwind mismatch
; 'call bar''s original unwind destination was 'C0', but after control flow
; linearization, its unwind destination incorrectly becomes 'C1'. We fix this by
; wrapping the call with a nested try-delegate that targets 'C0'.
; - Catch unwind mismatch
; If 'call foo' throws a foreign exception, it will not be caught by C1, and
; should be rethrown to the caller. But after control flow linearization, it
; will instead unwind to C0, an incorrect next EH pad. We wrap the whole
; try-catch with try-delegate that rethrows an exception to the caller to fix
; this.

; NOSORT-LABEL: test5
; NOSORT: try
; --- try-delegate starts (catch unwind mismatch)
; NOSORT    try
; NOSORT:     try
; NOSORT:       call  foo
; --- try-delegate starts (call unwind mismatch)
; NOSORT:       try
; NOSORT:         call  bar
; NOSORT:       delegate    2     # label/catch{{[0-9]+}}: down to catch[[C0:[0-9]+]]
; --- try-delegate ends (call unwind mismatch)
; NOSORT:     catch   {{.*}}      # catch[[C1:[0-9]+]]:
; NOSORT:     end_try
; NOSORT:   delegate    1         # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (catch unwind mismatch)
; NOSORT: catch   {{.*}}          # catch[[C0]]:
; NOSORT: end_try
; NOSORT: return

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

; 'call bar' and 'call baz''s original unwind destination was the caller, but
; after control flow linearization, their unwind destination incorrectly becomes
; 'C0'. We fix this by wrapping the calls with a nested try-delegate that
; rethrows exceptions to the caller.

; And the return value of 'baz' should NOT be stackified because the BB is split
; during fixing unwind mismatches.

; NOSORT-LABEL: test6
; NOSORT: try
; NOSORT:   call  foo
; --- try-delegate starts (call unwind mismatch)
; NOSORT:   try
; NOSORT:     call  bar
; NOSORT:     call  $[[RET:[0-9]+]]=, baz
; NOSORT-NOT: call  $push{{.*}}=, baz
; NOSORT:   delegate    1                     # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (call unwind mismatch)
; NOSORT:   call  nothrow, $[[RET]]
; NOSORT:   return
; NOSORT: catch   {{.*}}                      # catch[[C0:[0-9]+]]:
; NOSORT:   return
; NOSORT: end_try

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

; The same as test5, but we have one more call 'call @foo' in bb1 which unwinds
; to the caller. IN this case bb1 has two call unwind mismatches: 'call @foo'
; unwinds to the caller and 'call @bar' unwinds to catch C0.

; NOSORT-LABEL: test7
; NOSORT: try
; --- try-delegate starts (catch unwind mismatch)
; NOSORT    try
; NOSORT:     try
; NOSORT:       call  foo
; --- try-delegate starts (call unwind mismatch)
; NOSORT:       try
; NOSORT:         call  foo
; NOSORT:       delegate    3     # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (call unwind mismatch)
; --- try-delegate starts (call unwind mismatch)
; NOSORT:       try
; NOSORT:         call  bar
; NOSORT:       delegate    2     # label/catch{{[0-9]+}}: down to catch[[C0:[0-9]+]]
; --- try-delegate ends (call unwind mismatch)
; NOSORT:     catch   {{.*}}      # catch[[C1:[0-9]+]]:
; NOSORT:     end_try
; NOSORT:   delegate    1         # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (catch unwind mismatch)
; NOSORT: catch   {{.*}}        # catch[[C0]]:
; NOSORT: end_try
; NOSORT: return

define void @test7() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %catch.dispatch0

bb1:                                              ; preds = %bb0
  call void @foo()
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

; Similar situation as @test6. Here 'call @qux''s original unwind destination
; was the caller, but after control flow linearization, their unwind destination
; incorrectly becomes 'C0' within the function. We fix this by wrapping the call
; with a nested try-delegate that rethrows the exception to the caller.

; Because 'call @qux' pops an argument pushed by 'i32.const 5' from stack, the
; nested 'try' should be placed before `i32.const 5', not between 'i32.const 5'
; and 'call @qux'.

; NOSORT-LABEL: test8
; NOSORT: try       i32
; NOSORT:   call  foo
; --- try-delegate starts (call unwind mismatch)
; NOSORT:   try
; NOSORT:     i32.const  $push{{[0-9]+}}=, 5
; NOSORT:     call  ${{[0-9]+}}=, qux
; NOSORT:   delegate    1                     # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (call unwind mismatch)
; NOSORT:   return
; NOSORT: catch   {{.*}}                      # catch[[C0:[0-9]+]]:
; NOSORT:   return
; NOSORT: end_try

define i32 @test8() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %catch.dispatch0

bb1:                                              ; preds = %bb0
  %0 = call i32 @qux(i32 5)
  ret i32 %0

catch.dispatch0:                                  ; preds = %bb0
  %1 = catchswitch within none [label %catch.start0] unwind to caller

catch.start0:                                     ; preds = %catch.dispatch0
  %2 = catchpad within %1 [i8* null]
  %3 = call i8* @llvm.wasm.get.exception(token %2)
  %j = call i32 @llvm.wasm.get.ehselector(token %2)
  catchret from %2 to label %try.cont

try.cont:                                         ; preds = %catch.start0
  ret i32 0
}

; Tests the case when TEE stackifies a register in RegStackify but it gets
; unstackified in fixCallUnwindMismatches in CFGStackify.

; NOSORT-LOCALS-LABEL: test9
define void @test9(i32 %x) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %catch.dispatch0

bb1:                                              ; preds = %bb0
  %t = add i32 %x, 4
  ; This %addr is used in multiple places, so tee is introduced in RegStackify,
  ; which stackifies the use of %addr in store instruction. A tee has two dest
  ; registers, the first of which is stackified and the second is not.
  ; But when we introduce a nested try-delegate in fixCallUnwindMismatches in
  ; CFGStackify, it is possible that we end up unstackifying the first dest
  ; register. In that case, we convert that tee into a copy.
  %addr = inttoptr i32 %t to i32*
  %load = load i32, i32* %addr
  %call = call i32 @baz()
  %add = add i32 %load, %call
  store i32 %add, i32* %addr
  ret void
; NOSORT-LOCALS:       i32.add
; NOSORT-LOCALS-NOT:   local.tee
; NOSORT-LOCALS-NEXT:  local.set

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

; We have two call unwind unwind mismatches:
; - A may-throw instruction unwinds to an incorrect EH pad after linearizing the
;   CFG, when it is supposed to unwind to another EH pad.
; - A may-throw instruction unwinds to an incorrect EH pad after linearizing the
;   CFG, when it is supposed to unwind to the caller.
; We also have a catch unwind mismatch: If an exception is not caught by the
; first catch because it is a non-C++ exception, it shouldn't unwind to the next
; catch, but it should unwind to the caller.

; NOSORT-LABEL: test10
; NOSORT: try
; --- try-delegate starts (catch unwind mismatch)
; NOSORT:   try
; NOSORT:     try
; NOSORT:       call  foo
; --- try-delegate starts (call unwind mismatch)
; NOSORT:       try
; NOSORT:         call  bar
; NOSORT:       delegate    2            # label/catch{{[0-9]+}}: down to catch[[C0:[0-9]+]]
; --- try-delegate ends (call unwind mismatch)
; NOSORT:     catch
; NOSORT:       call  {{.*}} __cxa_begin_catch
; --- try-delegate starts (call unwind mismatch)
; NOSORT:       try
; NOSORT:         call  __cxa_end_catch
; NOSORT:       delegate    3            # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (call unwind mismatch)
; NOSORT:     end_try
; NOSORT:   delegate    1                # label/catch{{[0-9]+}}: to caller
; --- try-delegate ends (catch unwind mismatch)
; NOSORT: catch  {{.*}}                  # catch[[C0]]:
; NOSORT:   call  {{.*}} __cxa_begin_catch
; NOSORT:   call  __cxa_end_catch
; NOSORT: end_try
; NOSORT: return

define void @test10() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
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

; NOOPT-LABEL: test11
; NOOPT: block
; NOOPT:   try
; NOOPT:     call      foo
; NOOPT:   catch
; NOOPT:   end_try
; NOOPT:   call      foo
; NOOPT: end_block
; NOOPT: return
define void @test11(i32 %arg) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
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

; Intrinsics like memcpy, memmove, and memset don't throw and are lowered into
; calls to external symbols (not global addresses) in instruction selection,
; which will be eventually lowered to library function calls.
; Because this test runs with -wasm-disable-ehpad-sort, these library calls in
; invoke.cont BB fall within try~end_try, but they shouldn't cause crashes or
; unwinding destination mismatches in CFGStackify.

; NOSORT-LABEL: test12
; NOSORT: try
; NOSORT:   call  foo
; NOSORT:   call {{.*}} memcpy
; NOSORT:   call {{.*}} memmove
; NOSORT:   call {{.*}} memset
; NOSORT:   return
; NOSORT: catch_all
; NOSORT:   rethrow 0
; NOSORT: end_try
define void @test12(i8* %a, i8* %b) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %o = alloca %class.Object, align 1
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 100, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %a, i8* %b, i32 100, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %a, i8 0, i32 100, i1 false)
  %call = call %class.Object* @_ZN6ObjectD2Ev(%class.Object* %o)
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  %call2 = call %class.Object* @_ZN6ObjectD2Ev(%class.Object* %o) [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

; Tests if 'try' marker is placed correctly. In this test, 'try' should be
; placed before the call to 'nothrow_i32' and not between the call to
; 'nothrow_i32' and 'fun', because the return value of 'nothrow_i32' is
; stackified and pushed onto the stack to be consumed by the call to 'fun'.

; CHECK-LABEL: test13
; CHECK: try
; CHECK: call      $push{{.*}}=, nothrow_i32
; CHECK: call      fun, $pop{{.*}}
define void @test13() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %call = call i32 @nothrow_i32()
  invoke void @fun(i32 %call)
          to label %invoke.cont unwind label %terminate

invoke.cont:                                      ; preds = %entry
  ret void

terminate:                                        ; preds = %entry
  %0 = cleanuppad within none []
  call void @_ZSt9terminatev() [ "funclet"(token %0) ]
  unreachable
}

; This crashed on debug mode (= when NDEBUG is not defined) when the logic for
; computing the innermost region was not correct, in which a loop region
; contains an exception region. This should pass CFGSort without crashing.
define void @test14() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %e = alloca %class.MyClass, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 9
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  invoke void @quux(i32 %i.0)
          to label %for.inc unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %for.body
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast ({ i8*, i8* }* @_ZTI7MyClass to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast ({ i8*, i8* }* @_ZTI7MyClass to i8*))
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  %5 = call i8* @__cxa_get_exception_ptr(i8* %2) [ "funclet"(token %1) ]
  %6 = bitcast i8* %5 to %class.MyClass*
  %call = call %class.MyClass* @_ZN7MyClassC2ERKS_(%class.MyClass* %e, %class.MyClass* dereferenceable(4) %6) [ "funclet"(token %1) ]
  %7 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  %x = getelementptr inbounds %class.MyClass, %class.MyClass* %e, i32 0, i32 0
  %8 = load i32, i32* %x, align 4
  invoke void @quux(i32 %8) [ "funclet"(token %1) ]
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %catch
  %call3 = call %class.MyClass* @_ZN7MyClassD2Ev(%class.MyClass* %e) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %for.inc

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow() [ "funclet"(token %1) ]
  unreachable

for.inc:                                          ; preds = %invoke.cont2, %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

ehcleanup:                                        ; preds = %catch
  %9 = cleanuppad within %1 []
  %call4 = call %class.MyClass* @_ZN7MyClassD2Ev(%class.MyClass* %e) [ "funclet"(token %9) ]
  invoke void @__cxa_end_catch() [ "funclet"(token %9) ]
          to label %invoke.cont6 unwind label %terminate7

invoke.cont6:                                     ; preds = %ehcleanup
  cleanupret from %9 unwind to caller

for.end:                                          ; preds = %for.cond
  ret void

terminate7:                                       ; preds = %ehcleanup
  %10 = cleanuppad within %9 []
  call void @_ZSt9terminatev() [ "funclet"(token %10) ]
  unreachable
}

; Tests if CFGStackify's removeUnnecessaryInstrs() removes unnecessary branches
; correctly. The code is in the form below, where 'br' is unnecessary because
; after running the 'try' body the control flow will fall through to bb2 anyway.

; bb0:
;   try
;     ...
;     br bb2      <- Not necessary
; bb1 (ehpad):
;   catch
;     ...
; bb2:            <- Continuation BB
;   end
; CHECK-LABEL: test15
define void @test15(i32 %n) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %for.body unwind label %catch.dispatch

for.body:                                         ; preds = %for.end, %entry
  %i = phi i32 [ %inc, %for.end ], [ 0, %entry ]
  invoke void @foo()
          to label %for.end unwind label %catch.dispatch

; Before going to CFGStackify, this BB will have a conditional branch followed
; by an unconditional branch. CFGStackify should remove only the unconditional
; one.
for.end:                                          ; preds = %for.body
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %try.cont, label %for.body
; CHECK: br_if
; CHECK-NOT: br
; CHECK: end_loop
; CHECK: catch

catch.dispatch:                                   ; preds = %for.body, %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.start, %for.end
  ret void
}

; void foo();
; void test16() {
;   try {
;     foo();
;     try {
;       foo();
;     } catch (...) {
;     }
;   } catch (...) {
;   }
; }
;
; This tests whether the 'br' can be removed in code in the form as follows.
; Here 'br' is inside an inner try, whose 'end' is in another EH pad. In this
; case, after running an inner try body, the control flow should fall through to
; bb3, so the 'br' in the code is unnecessary.

; bb0:
;   try
;     try
;       ...
;       br bb3      <- Not necessary
; bb1:
;     catch
; bb2:
;     end_try
;   catch
;     ...
; bb3:            <- Continuation BB
;   end
;
; CHECK-LABEL: test16
define void @test16() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK: call foo
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %catch.dispatch3

; CHECK: call foo
; CHECK-NOT: br
invoke.cont:                                      ; preds = %entry
  invoke void @foo()
          to label %try.cont8 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont
  %0 = catchswitch within none [label %catch.start] unwind label %catch.dispatch3

; CHECK: catch
catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  invoke void @__cxa_end_catch() [ "funclet"(token %1) ]
          to label %invoke.cont2 unwind label %catch.dispatch3

catch.dispatch3:                                  ; preds = %catch.start, %catch.dispatch, %entry
  %5 = catchswitch within none [label %catch.start4] unwind to caller

catch.start4:                                     ; preds = %catch.dispatch3
  %6 = catchpad within %5 [i8* null]
  %7 = call i8* @llvm.wasm.get.exception(token %6)
  %8 = call i32 @llvm.wasm.get.ehselector(token %6)
  %9 = call i8* @__cxa_begin_catch(i8* %7) [ "funclet"(token %6) ]
  call void @__cxa_end_catch() [ "funclet"(token %6) ]
  catchret from %6 to label %try.cont8

try.cont8:                                        ; preds = %invoke.cont2, %catch.start4, %invoke.cont
  ret void

invoke.cont2:                                     ; preds = %catch.start
  catchret from %1 to label %try.cont8
}

; Here an exception is semantically contained in a loop. 'ehcleanup' BB belongs
; to the exception, but does not belong to the loop (because it does not have a
; path back to the loop header), and is placed after the loop latch block
; 'invoke.cont' intentionally. This tests if 'end_loop' marker is placed
; correctly not right after 'invoke.cont' part but after 'ehcleanup' part,
; NOSORT-LABEL: test17
; NOSORT: loop
; NOSORT: try
; NOSORT: end_try
; NOSORT: end_loop
define void @test17(i32 %n) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %invoke.cont, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %invoke.cont ]
  %tobool = icmp ne i32 %n.addr.0, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %dec = add nsw i32 %n.addr.0, -1
  invoke void @foo()
          to label %while.end unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %while.body
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  invoke void @__cxa_end_catch() [ "funclet"(token %1) ]
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %catch.start
  catchret from %1 to label %while.cond

ehcleanup:                                        ; preds = %catch.start
  %5 = cleanuppad within %1 []
  call void @_ZSt9terminatev() [ "funclet"(token %5) ]
  unreachable

while.end:                                        ; preds = %while.body, %while.cond
  ret void
}

; When the function return type is non-void and 'end' instructions are at the
; very end of a function, CFGStackify's fixEndsAtEndOfFunction function fixes
; the corresponding block/loop/try's type to match the function's return type.
; But when a `try`'s type is fixed, we should also check `end` instructions
; before its corresponding `catch_all`, because both `try` and `catch_all` body
; should satisfy the return type requirements.

; NOSORT-LABEL: test18
; NOSORT: try i32
; NOSORT: loop i32
; NOSORT: end_loop
; NOSORT: catch_all
; NOSORT: end_try
; NOSORT-NEXT: end_function
define i32 @test18(i32 %n) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %t = alloca %class.Object, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br label %for.body

for.body:                                         ; preds = %for.cond
  %div = sdiv i32 %n, 2
  %cmp1 = icmp eq i32 %i.0, %div
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %call = invoke i32 @baz()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %if.then
  %call2 = call %class.Object* @_ZN6ObjectD2Ev(%class.Object* %t)
  ret i32 %call

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

ehcleanup:                                        ; preds = %if.then
  %0 = cleanuppad within none []
  %call3 = call %class.Object* @_ZN6ObjectD2Ev(%class.Object* %t) [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

; This crashed when updating EHPadStack within fixCallUniwindMismatch had a bug.
; This should not crash and try-delegate has to be created around 'call @baz',
; because the initial TRY placement for 'call @quux' was done before 'call @baz'
; because 'call @baz''s return value is stackified.

; CHECK-LABEL: test19
; CHECK: try
; CHECK:   try
; CHECK:     call $[[RET:[0-9]+]]=, baz
; CHECK:   delegate  1
; CHECK:    call  quux, $[[RET]]
; CHECK: catch_all
; CHECK: end_try
define void @test19() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %call = call i32 @baz()
  invoke void @quux(i32 %call)
          to label %invoke.cont unwind label %ehcleanup

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  cleanupret from %0 unwind to caller

invoke.cont:                                      ; preds = %entry
  unreachable
}

; This tests if invalidated branch destinations after fixing catch unwind
; mismatches are correctly remapped. For example, we have this code and suppose
; we need to wrap this try-catch-end in this code with a try-delegate to fix a
; catch unwind mismatch:
  ; - Before:
; block
;   br (a)
;   try
;   catch
;   end_try
; end_block
;           <- (a)
;
; - After
; block
;   br (a)
;   try
;     try
;     catch
;     end_try
;           <- (a)
;   delegate
; end_block
;           <- (b)
; After adding a try-delegate, the 'br's destination BB, where (a) points,
; becomes invalid because it incorrectly branches into an inner scope. The
; destination should change to the BB where (b) points.

; NOSORT-LABEL: test20
; NOSORT: try
; NOSORT:   br_if   0
define void @test20(i1 %arg) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  br i1 %arg, label %bb0, label %dest

bb0:                                              ; preds = %entry
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

dest:                                             ; preds = %entry
  ret void

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

; The similar case with test20, but multiple consecutive delegates are
; generated:
; - Before:
; block
;   br (a)
;   try
;   catch
;   end_try
; end_block
;           <- (a)
;
; - After
; block
;   br (a)
;   try
;     ...
;     try
;       try
;       catch
;       end_try
;             <- (a)
;     delegate
;   delegate
; end_block
;           <- (b) The br destination should be remapped to here
;
; The test was reduced by bugpoint and should not crash in CFGStackify.
define void @test21() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  br i1 undef, label %if.then, label %if.end12

if.then:                                          ; preds = %entry
  invoke void @__cxa_throw(i8* null, i8* null, i8* null) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.then
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch.start
  invoke void @foo()
          to label %invoke.cont unwind label %catch.dispatch4

invoke.cont:                                      ; preds = %catchret.dest
  invoke void @__cxa_throw(i8* null, i8* null, i8* null) #1
          to label %unreachable unwind label %catch.dispatch4

catch.dispatch4:                                  ; preds = %invoke.cont, %catchret.dest
  %4 = catchswitch within none [label %catch.start5] unwind to caller

catch.start5:                                     ; preds = %catch.dispatch4
  %5 = catchpad within %4 [i8* bitcast (i8** @_ZTIi to i8*)]
  %6 = call i8* @llvm.wasm.get.exception(token %5)
  %7 = call i32 @llvm.wasm.get.ehselector(token %5)
  unreachable

if.end12:                                         ; preds = %entry
  invoke void @foo()
          to label %invoke.cont14 unwind label %catch.dispatch16

catch.dispatch16:                                 ; preds = %if.end12
  %8 = catchswitch within none [label %catch.start17] unwind label %ehcleanup

catch.start17:                                    ; preds = %catch.dispatch16
  %9 = catchpad within %8 [i8* bitcast (i8** @_ZTIi to i8*)]
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  %11 = call i32 @llvm.wasm.get.ehselector(token %9)
  br i1 undef, label %catch20, label %rethrow19

catch20:                                          ; preds = %catch.start17
  catchret from %9 to label %catchret.dest22

catchret.dest22:                                  ; preds = %catch20
  br label %try.cont23

rethrow19:                                        ; preds = %catch.start17
  invoke void @llvm.wasm.rethrow() #1 [ "funclet"(token %9) ]
          to label %unreachable unwind label %ehcleanup

try.cont23:                                       ; preds = %invoke.cont14, %catchret.dest22
  invoke void @foo()
          to label %invoke.cont24 unwind label %ehcleanup

invoke.cont24:                                    ; preds = %try.cont23
  ret void

invoke.cont14:                                    ; preds = %if.end12
  br label %try.cont23

ehcleanup:                                        ; preds = %try.cont23, %rethrow19, %catch.dispatch16
  %12 = cleanuppad within none []
  cleanupret from %12 unwind to caller

unreachable:                                      ; preds = %rethrow19, %invoke.cont, %if.then
  unreachable
}

; Regression test for WasmEHFuncInfo's reverse mapping bug. 'UnwindDestToSrc'
; should return a vector and not a single BB, which was incorrect.
; This was reduced by bugpoint and should not crash in CFGStackify.
define void @test22() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind label %ehcleanup22

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  invoke void @__cxa_throw(i8* null, i8* null, i8* null) #1 [ "funclet"(token %1) ]
          to label %unreachable unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %catch.start
  %4 = catchswitch within %1 [label %catch.start3] unwind label %ehcleanup

catch.start3:                                     ; preds = %catch.dispatch2
  %5 = catchpad within %4 [i8* bitcast (i8** @_ZTIi to i8*)]
  %6 = call i8* @llvm.wasm.get.exception(token %5)
  %7 = call i32 @llvm.wasm.get.ehselector(token %5)
  catchret from %5 to label %try.cont

try.cont:                                         ; preds = %catch.start3
  invoke void @foo() [ "funclet"(token %1) ]
          to label %invoke.cont8 unwind label %ehcleanup

invoke.cont8:                                     ; preds = %try.cont
  invoke void @__cxa_throw(i8* null, i8* null, i8* null) #1 [ "funclet"(token %1) ]
          to label %unreachable unwind label %catch.dispatch11

catch.dispatch11:                                 ; preds = %invoke.cont8
  %8 = catchswitch within %1 [label %catch.start12] unwind label %ehcleanup

catch.start12:                                    ; preds = %catch.dispatch11
  %9 = catchpad within %8 [i8* bitcast (i8** @_ZTIi to i8*)]
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  %11 = call i32 @llvm.wasm.get.ehselector(token %9)
  unreachable

invoke.cont:                                      ; preds = %entry
  unreachable

ehcleanup:                                        ; preds = %catch.dispatch11, %try.cont, %catch.dispatch2
  %12 = cleanuppad within %1 []
  cleanupret from %12 unwind label %ehcleanup22

ehcleanup22:                                      ; preds = %ehcleanup, %catch.dispatch
  %13 = cleanuppad within none []
  cleanupret from %13 unwind to caller

unreachable:                                      ; preds = %invoke.cont8, %catch.start
  unreachable
}

; void test23() {
;   try {
;     try {
;       throw 0;
;     } catch (int) {
;     }
;   } catch (int) {
;   }
; }
;
; Regression test for a WebAssemblyException grouping bug. After catchswitches
; are removed, EH pad catch.start2 is dominated by catch.start, but because
; catch.start2 is the unwind destination of catch.start, it should not be
; included in catch.start's exception. Also, after we take catch.start2's
; exception out of catch.start's exception, we have to take out try.cont8 out of
; catch.start's exception, because it has a predecessor in catch.start2.
define void @test23() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %exception = call i8* @__cxa_allocate_exception(i32 4) #0
  %0 = bitcast i8* %exception to i32*
  store i32 0, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %catch.start] unwind label %catch.dispatch1

catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* bitcast (i8** @_ZTIi to i8*)]
  %3 = call i8* @llvm.wasm.get.exception(token %2)
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  %5 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #0
  %matches = icmp eq i32 %4, %5
  br i1 %matches, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  %6 = call i8* @__cxa_begin_catch(i8* %3) #0 [ "funclet"(token %2) ]
  %7 = bitcast i8* %6 to i32*
  %8 = load i32, i32* %7, align 4
  call void @__cxa_end_catch() #0 [ "funclet"(token %2) ]
  catchret from %2 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

rethrow:                                          ; preds = %catch.start
  invoke void @llvm.wasm.rethrow() #1 [ "funclet"(token %2) ]
          to label %unreachable unwind label %catch.dispatch1

catch.dispatch1:                                  ; preds = %rethrow, %catch.dispatch
  %9 = catchswitch within none [label %catch.start2] unwind to caller

catch.start2:                                     ; preds = %catch.dispatch1
  %10 = catchpad within %9 [i8* bitcast (i8** @_ZTIi to i8*)]
  %11 = call i8* @llvm.wasm.get.exception(token %10)
  %12 = call i32 @llvm.wasm.get.ehselector(token %10)
  %13 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #0
  %matches3 = icmp eq i32 %12, %13
  br i1 %matches3, label %catch5, label %rethrow4

catch5:                                           ; preds = %catch.start2
  %14 = call i8* @__cxa_begin_catch(i8* %11) #0 [ "funclet"(token %10) ]
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4
  call void @__cxa_end_catch() #0 [ "funclet"(token %10) ]
  catchret from %10 to label %catchret.dest7

catchret.dest7:                                   ; preds = %catch5
  br label %try.cont8

rethrow4:                                         ; preds = %catch.start2
  call void @llvm.wasm.rethrow() #1 [ "funclet"(token %10) ]
  unreachable

try.cont8:                                        ; preds = %try.cont, %catchret.dest7
  ret void

try.cont:                                         ; preds = %catchret.dest
  br label %try.cont8

unreachable:                                      ; preds = %rethrow, %entry
  unreachable
}

; Test for WebAssemblyException grouping. This test is hand-modified to generate
; this structure:
; catch.start dominates catch.start4 and catch.start4 dominates catch.start12,
; so the after dominator-based grouping, we end up with:
; catch.start's exception > catch4.start's exception > catch12.start's exception
; (> here represents subexception relationship)
;
; But the unwind destination chain is catch.start -> catch.start4 ->
; catch.start12. So all these subexception relationship should be deconstructed.
; We have to make sure to take out catch.start4's exception out of catch.start's
; exception first, before taking out catch.start12's exception out of
; catch.start4's exception; otherwise we end up with an incorrect relationship
; of catch.start's exception > catch.start12's exception.
define void @test24() personality i8* bitcast (i32 (...)*
@__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %entry
  invoke void @foo()
          to label %invoke.cont1 unwind label %catch.dispatch

invoke.cont1:                                     ; preds = %invoke.cont
  invoke void @foo()
          to label %try.cont18 unwind label %catch.dispatch

catch.dispatch11:                                 ; preds = %rethrow6, %catch.dispatch3
  %0 = catchswitch within none [label %catch.start12] unwind to caller

catch.start12:                                    ; preds = %catch.dispatch11
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #0
  %matches13 = icmp eq i32 %3, %4
  br i1 %matches13, label %catch15, label %rethrow14

catch15:                                          ; preds = %catch.start12
  %5 = call i8* @__cxa_begin_catch(i8* %2) #0 [ "funclet"(token %1) ]
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  call void @__cxa_end_catch() #0 [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont18

rethrow14:                                        ; preds = %catch.start12
  call void @llvm.wasm.rethrow() #1 [ "funclet"(token %1) ]
  unreachable

catch.dispatch3:                                  ; preds = %rethrow, %catch.dispatch
  %8 = catchswitch within none [label %catch.start4] unwind label %catch.dispatch11

catch.start4:                                     ; preds = %catch.dispatch3
  %9 = catchpad within %8 [i8* bitcast (i8** @_ZTIi to i8*)]
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  %11 = call i32 @llvm.wasm.get.ehselector(token %9)
  %12 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #0
  %matches5 = icmp eq i32 %11, %12
  br i1 %matches5, label %catch7, label %rethrow6

catch7:                                           ; preds = %catch.start4
  %13 = call i8* @__cxa_begin_catch(i8* %10) #0 [ "funclet"(token %9) ]
  %14 = bitcast i8* %13 to i32*
  %15 = load i32, i32* %14, align 4
  call void @__cxa_end_catch() #0 [ "funclet"(token %9) ]
  catchret from %9 to label %try.cont18

rethrow6:                                         ; preds = %catch.start4
  invoke void @llvm.wasm.rethrow() #1 [ "funclet"(token %9) ]
          to label %unreachable unwind label %catch.dispatch11

catch.dispatch:                                   ; preds = %invoke.cont1, %invoke.cont, %entry
  %16 = catchswitch within none [label %catch.start] unwind label %catch.dispatch3

catch.start:                                      ; preds = %catch.dispatch
  %17 = catchpad within %16 [i8* bitcast (i8** @_ZTIi to i8*)]
  %18 = call i8* @llvm.wasm.get.exception(token %17)
  %19 = call i32 @llvm.wasm.get.ehselector(token %17)
  %20 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #0
  %matches = icmp eq i32 %19, %20
  br i1 %matches, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  %21 = call i8* @__cxa_begin_catch(i8* %18) #0 [ "funclet"(token %17) ]
  %22 = bitcast i8* %21 to i32*
  %23 = load i32, i32* %22, align 4
  call void @__cxa_end_catch() #0 [ "funclet"(token %17) ]
  catchret from %17 to label %try.cont18

rethrow:                                          ; preds = %catch.start
  invoke void @llvm.wasm.rethrow() #1 [ "funclet"(token %17) ]
          to label %unreachable unwind label %catch.dispatch3

try.cont18:                                       ; preds = %catch, %catch7, %catch15, %invoke.cont1
  ret void

unreachable:                                      ; preds = %rethrow, %rethrow6
  unreachable
}

; void test25() {
;   try {
;     try {
;       throw 0;
;     } catch (int) { // (a)
;     }
;   } catch (int) {   // (b)
;   }
;   try {
;     foo();
;   } catch (int) {   // (c)
;   }
; }
;
; Regression test for an ExceptionInfo grouping bug. Because the first (inner)
; try always throws, both EH pads (b) (catch.start2) and (c) (catch.start10) are
; dominated by EH pad (a) (catch.start), even though they are not semantically
; contained in (a)'s exception. Because (a)'s unwind destination is (b), (b)'s
; exception is taken out of (a)'s. But because (c) is reachable from (b), we
; should make sure to take out (c)'s exception out of (a)'s exception too.
define void @test25() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  %exception = call i8* @__cxa_allocate_exception(i32 4) #1
  %0 = bitcast i8* %exception to i32*
  store i32 0, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %catch.start] unwind label %catch.dispatch1

catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* bitcast (i8** @_ZTIi to i8*)]
  %3 = call i8* @llvm.wasm.get.exception(token %2)
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  %5 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #1
  %matches = icmp eq i32 %4, %5
  br i1 %matches, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  %6 = call i8* @__cxa_begin_catch(i8* %3) #1 [ "funclet"(token %2) ]
  %7 = bitcast i8* %6 to i32*
  %8 = load i32, i32* %7, align 4
  call void @__cxa_end_catch() #1 [ "funclet"(token %2) ]
  catchret from %2 to label %try.cont8

rethrow:                                          ; preds = %catch.start
  invoke void @llvm.wasm.rethrow() #3 [ "funclet"(token %2) ]
          to label %unreachable unwind label %catch.dispatch1

catch.dispatch1:                                  ; preds = %rethrow, %catch.dispatch
  %9 = catchswitch within none [label %catch.start2] unwind to caller

catch.start2:                                     ; preds = %catch.dispatch1
  %10 = catchpad within %9 [i8* bitcast (i8** @_ZTIi to i8*)]
  %11 = call i8* @llvm.wasm.get.exception(token %10)
  %12 = call i32 @llvm.wasm.get.ehselector(token %10)
  %13 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #1
  %matches3 = icmp eq i32 %12, %13
  br i1 %matches3, label %catch5, label %rethrow4

catch5:                                           ; preds = %catch.start2
  %14 = call i8* @__cxa_begin_catch(i8* %11) #1 [ "funclet"(token %10) ]
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4
  call void @__cxa_end_catch() #1 [ "funclet"(token %10) ]
  catchret from %10 to label %try.cont8

rethrow4:                                         ; preds = %catch.start2
  call void @llvm.wasm.rethrow() #3 [ "funclet"(token %10) ]
  unreachable

try.cont8:                                        ; preds = %catch, %catch5
  invoke void @foo()
          to label %try.cont16 unwind label %catch.dispatch9

catch.dispatch9:                                  ; preds = %try.cont8
  %17 = catchswitch within none [label %catch.start10] unwind to caller

catch.start10:                                    ; preds = %catch.dispatch9
  %18 = catchpad within %17 [i8* bitcast (i8** @_ZTIi to i8*)]
  %19 = call i8* @llvm.wasm.get.exception(token %18)
  %20 = call i32 @llvm.wasm.get.ehselector(token %18)
  %21 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #1
  %matches11 = icmp eq i32 %20, %21
  br i1 %matches11, label %catch13, label %rethrow12

catch13:                                          ; preds = %catch.start10
  %22 = call i8* @__cxa_begin_catch(i8* %19) #1 [ "funclet"(token %18) ]
  %23 = bitcast i8* %22 to i32*
  %24 = load i32, i32* %23, align 4
  call void @__cxa_end_catch() #1 [ "funclet"(token %18) ]
  catchret from %18 to label %try.cont16

rethrow12:                                        ; preds = %catch.start10
  call void @llvm.wasm.rethrow() #3 [ "funclet"(token %18) ]
  unreachable

try.cont16:                                       ; preds = %try.cont8, %catch13
  ret void

unreachable:                                      ; preds = %rethrow, %entry
  unreachable
}

; Check if the unwind destination mismatch stats are correct
; NOSORT: 23 wasm-cfg-stackify    - Number of call unwind mismatches found
; NOSORT:  4 wasm-cfg-stackify    - Number of catch unwind mismatches found

declare void @foo()
declare void @bar()
declare i32 @baz()
declare i32 @qux(i32)
declare void @quux(i32)
declare void @fun(i32)
; Function Attrs: nounwind
declare void @nothrow(i32) #0
; Function Attrs: nounwind
declare i32 @nothrow_i32() #0

; Function Attrs: nounwind
declare %class.Object* @_ZN6ObjectD2Ev(%class.Object* returned) #0
@_ZTI7MyClass = external constant { i8*, i8* }, align 4
; Function Attrs: nounwind
declare %class.MyClass* @_ZN7MyClassD2Ev(%class.MyClass* returned) #0
; Function Attrs: nounwind
declare %class.MyClass* @_ZN7MyClassC2ERKS_(%class.MyClass* returned, %class.MyClass* dereferenceable(4)) #0

declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nounwind
declare i8* @llvm.wasm.get.exception(token) #0
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #0
declare i8* @__cxa_allocate_exception(i32) #0
declare void @__cxa_throw(i8*, i8*, i8*)
; Function Attrs: noreturn
declare void @llvm.wasm.rethrow() #1
; Function Attrs: nounwind
declare i32 @llvm.eh.typeid.for(i8*) #0

declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare i8* @__cxa_get_exception_ptr(i8*)
declare void @_ZSt9terminatev()
; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg) #0
; Function Attrs: nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1 immarg) #0
; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1 immarg) #0

attributes #0 = { nounwind }
attributes #1 = { noreturn }
