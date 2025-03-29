; RUN: opt < %s -win-eh-prepare -demote-catchswitch-only -wasm-eh-prepare -S | FileCheck %s
; RUN: opt < %s -win-eh-prepare -demote-catchswitch-only -wasm-eh-prepare -S --mattr=+atomics,+bulk-memory | FileCheck %s
; RUN: opt < %s -passes='win-eh-prepare<demote-catchswitch-only>,wasm-eh-prepare' -S | FileCheck %s
; RUN: opt < %s -passes='win-eh-prepare<demote-catchswitch-only>,wasm-eh-prepare' -S --mattr=+atomics,+bulk-memory | FileCheck %s
; RUN: llc < %s -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -stop-after=wasm-eh-prepare | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: @__wasm_lpad_context = external thread_local global { i32, ptr, i32 }

@_ZTIi = external constant ptr
%struct.Temp = type { i8 }

; A single 'catch (int)' clause.
; A wasm.catch() call, wasm.lsda() call, and personality call to generate a
; selector should all be genereated after the catchpad.
;
; void foo();
; void test0() {
;   try {
;     foo();
;   } catch (int) {
;   }
; }
define void @test0() personality ptr @__gxx_wasm_personality_v0 {
; CHECK-LABEL: @test0()
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @_ZTIi]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %rethrow
; CHECK: catch.start:
; CHECK-NEXT:   %[[CATCHPAD:.*]] = catchpad
; CHECK-NEXT:   %[[EXN:.*]] = call ptr @llvm.wasm.catch(i32 0)
; CHECK-NEXT:   call void @llvm.wasm.landingpad.index(token %[[CATCHPAD]], i32 0)
; CHECK-NEXT:   store i32 0, ptr @__wasm_lpad_context
; CHECK-NEXT:   %[[LSDA:.*]] = call ptr @llvm.wasm.lsda()
; CHECK-NEXT:   store ptr %[[LSDA]], ptr getelementptr inbounds ({ i32, ptr, i32 }, ptr @__wasm_lpad_context, i32 0, i32 1)
; CHECK-NEXT:   call i32 @_Unwind_CallPersonality(ptr %[[EXN]]) {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
; CHECK-NEXT:   %[[SELECTOR:.*]] = load i32, ptr getelementptr inbounds ({ i32, ptr, i32 }, ptr @__wasm_lpad_context, i32 0, i32 2)
; CHECK:   icmp eq i32 %[[SELECTOR]]

catch:                                            ; preds = %catch.start
  %5 = call ptr @__cxa_begin_catch(ptr %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont
; CHECK: catch:
; CHECK-NEXT:  call ptr @__cxa_begin_catch(ptr %[[EXN]])

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont:                                         ; preds = %entry, %catch
  ret void
}

; Two try-catches.
; For the catchpad with a single 'catch (...)', only a wasm.catch() call should
; be generated after the catchpad; wasm.landingpad.index() and personality call
; should NOT be generated. For the other catchpad, the argument of
; wasm.landingpad.index() should be not 1 but 0.
;
; void foo();
; void test1() {
;   try {
;     foo();
;   } catch (...) {
;   }
;   try {
;     foo();
;   } catch (int) {
;   }
; }
define void @test1() personality ptr @__gxx_wasm_personality_v0 {
; CHECK-LABEL: @test1()
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call ptr @__cxa_begin_catch(ptr %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont
; CHECK: catch.start:
; CHECK-NEXT:   catchpad within %0 [ptr null]
; CHECK-NOT:   call void @llvm.wasm.landingpad.index
; CHECK-NOT:   store {{.*}} @__wasm_lpad_context
; CHECK-NOT:   call ptr @llvm.wasm.lsda()
; CHECK-NOT:   call i32 @_Unwind_CallPersonality
; CHECK-NOT:   load {{.*}} @__wasm_lpad_context

try.cont:                                         ; preds = %entry, %catch.start
  invoke void @foo()
          to label %try.cont7 unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %try.cont
  %5 = catchswitch within none [label %catch.start3] unwind to caller

catch.start3:                                     ; preds = %catch.dispatch2
  %6 = catchpad within %5 [ptr @_ZTIi]
  %7 = call ptr @llvm.wasm.get.exception(token %6)
  %8 = call i32 @llvm.wasm.get.ehselector(token %6)
  %9 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %8, %9
  br i1 %matches, label %catch4, label %rethrow
; CHECK: catch.start3:
; CHECK:   call void @llvm.wasm.landingpad.index(token %{{.+}}, i32 0)

catch4:                                           ; preds = %catch.start3
  %10 = call ptr @__cxa_begin_catch(ptr %7) [ "funclet"(token %6) ]
  call void @__cxa_end_catch() [ "funclet"(token %6) ]
  catchret from %6 to label %try.cont7

rethrow:                                          ; preds = %catch.start3
  call void @llvm.wasm.rethrow() [ "funclet"(token %6) ]
  unreachable

try.cont7:                                        ; preds = %try.cont, %catch4
  ret void
}

; PHI demotion test. Only the phi before catchswitch should be demoted; the phi
; before cleanuppad should NOT.
;
; void foo();
; int bar(int) noexcept;
; struct Temp {
;   ~Temp() {}
; };
;
; void test2() {
;   int num;
;   try {
;     Temp t;
;     num = 1;
;     foo();
;     num = 2;
;     foo();
;   } catch (...) {
;     bar(num);
;   }
;   try {
;     foo();
;     num = 1;
;     foo();
;     num = 2;
;   } catch (...) {
;     bar(num);
;   }
; }
define void @test2() personality ptr @__gxx_wasm_personality_v0 {
; CHECK-LABEL: @test2
entry:
  %t = alloca %struct.Temp, align 1
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @foo()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %invoke.cont
  %call = call ptr @_ZN4TempD2Ev(ptr %t)
  br label %try.cont

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %num.0 = phi i32 [ 2, %invoke.cont ], [ 1, %entry ]
  %0 = cleanuppad within none []
  %call2 = call ptr @_ZN4TempD2Ev(ptr %t) [ "funclet"(token %0) ]
  cleanupret from %0 unwind label %catch.dispatch
; CHECK: ehcleanup:
; CHECK-NEXT:   = phi

catch.dispatch:                                   ; preds = %ehcleanup
  %1 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr null]
  %3 = call ptr @llvm.wasm.get.exception(token %2)
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  %5 = call ptr @__cxa_begin_catch(ptr %3) [ "funclet"(token %2) ]
  call void @bar(i32 %num.0) [ "funclet"(token %2) ]
  call void @__cxa_end_catch() [ "funclet"(token %2) ]
  catchret from %2 to label %try.cont

try.cont:                                         ; preds = %catch.start, %invoke.cont1
  invoke void @foo()
          to label %invoke.cont3 unwind label %catch.dispatch5

invoke.cont3:                                     ; preds = %try.cont
  invoke void @foo()
          to label %try.cont10 unwind label %catch.dispatch5

catch.dispatch5:                                  ; preds = %invoke.cont3, %try.cont
  %num.1 = phi i32 [ 2, %invoke.cont3 ], [ 1, %try.cont ]
  %6 = catchswitch within none [label %catch.start6] unwind to caller
; CHECK: catch.dispatch5:
; CHECK-NOT:   = phi

catch.start6:                                     ; preds = %catch.dispatch5
  %7 = catchpad within %6 [ptr null]
  %8 = call ptr @llvm.wasm.get.exception(token %7)
  %9 = call i32 @llvm.wasm.get.ehselector(token %7)
  %10 = call ptr @__cxa_begin_catch(ptr %8) [ "funclet"(token %7) ]
  call void @bar(i32 %num.1) [ "funclet"(token %7) ]
  call void @__cxa_end_catch() [ "funclet"(token %7) ]
  catchret from %7 to label %try.cont10

try.cont10:                                       ; preds = %invoke.cont3, %catch.start6
  ret void
}

; Tests if instructions after a call to @llvm.wasm.throw are deleted and the
; BB's dead children are deleted.

; CHECK-LABEL: @test3
define i32 @test3(i1 %b, ptr %p) {
entry:
  br i1 %b, label %bb.true, label %bb.false

; CHECK:      bb.true:
; CHECK-NEXT:   call void @llvm.wasm.throw(i32 0, ptr %p)
; CHECK-NEXT:   unreachable
bb.true:                                          ; preds = %entry
  call void @llvm.wasm.throw(i32 0, ptr %p)
  br label %bb.true.0

; CHECK-NOT:  bb.true.0
bb.true.0:                                        ; preds = %bb.true
  br label %merge

bb.false:                                         ; preds = %entry
  br label %merge

; CHECK:      merge
merge:                                            ; preds = %bb.true.0, %bb.false
  ret i32 0
}

declare void @foo()
declare void @bar(i32)
declare ptr @_ZN4TempD2Ev(ptr returned)
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nounwind
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #0
; Function Attrs: nounwind
declare i32 @llvm.eh.typeid.for(ptr) #0
; Function Attrs: noreturn
declare void @llvm.wasm.throw(i32, ptr) #1
; Function Attrs: noreturn
declare void @llvm.wasm.rethrow() #1
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @_ZSt9terminatev()

attributes #0 = { nounwind }
attributes #1 = { noreturn }

; CHECK-DAG: declare void @llvm.wasm.landingpad.index(token, i32 immarg)
; CHECK-DAG: declare ptr @llvm.wasm.lsda()
; CHECK-DAG: declare i32 @_Unwind_CallPersonality(ptr)
