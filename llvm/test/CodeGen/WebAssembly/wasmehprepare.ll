; RUN: opt < %s -winehprepare -demote-catchswitch-only -wasmehprepare -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: @__wasm_lpad_context = external global { i32, i8*, i32 }

@_ZTIi = external constant i8*
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
define void @test0() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test0()
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
; CHECK: catch.start:
; CHECK-NEXT:   %[[CATCHPAD:.*]] = catchpad
; CHECK-NEXT:   %[[EXN:.*]] = call i8* @llvm.wasm.catch(i32 0)
; CHECK-NEXT:   call void @llvm.wasm.landingpad.index(token %[[CATCHPAD]], i32 0)
; CHECK-NEXT:   store i32 0, i32* getelementptr inbounds ({ i32, i8*, i32 }, { i32, i8*, i32 }* @__wasm_lpad_context, i32 0, i32 0)
; CHECK-NEXT:   %[[LSDA:.*]] = call i8* @llvm.wasm.lsda()
; CHECK-NEXT:   store i8* %[[LSDA]], i8** getelementptr inbounds ({ i32, i8*, i32 }, { i32, i8*, i32 }* @__wasm_lpad_context, i32 0, i32 1)
; CHECK-NEXT:   call i32 @_Unwind_CallPersonality(i8* %[[EXN]]) {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
; CHECK-NEXT:   %[[SELECTOR:.*]] = load i32, i32* getelementptr inbounds ({ i32, i8*, i32 }, { i32, i8*, i32 }* @__wasm_lpad_context, i32 0, i32 2)
; CHECK:   icmp eq i32 %[[SELECTOR]]

catch:                                            ; preds = %catch.start
  %5 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont
; CHECK: catch:
; CHECK-NEXT:  call i8* @__cxa_begin_catch(i8* %[[EXN]])

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
define void @test1() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test1()
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
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont
; CHECK: catch.start:
; CHECK-NEXT:   catchpad within %0 [i8* null]
; CHECK-NOT:   call void @llvm.wasm.landingpad.index
; CHECK-NOT:   store {{.*}} @__wasm_lpad_context
; CHECK-NOT:   call i8* @llvm.wasm.lsda()
; CHECK-NOT:   call i32 @_Unwind_CallPersonality
; CHECK-NOT:   load {{.*}} @__wasm_lpad_context

try.cont:                                         ; preds = %entry, %catch.start
  invoke void @foo()
          to label %try.cont7 unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %try.cont
  %5 = catchswitch within none [label %catch.start3] unwind to caller

catch.start3:                                     ; preds = %catch.dispatch2
  %6 = catchpad within %5 [i8* bitcast (i8** @_ZTIi to i8*)]
  %7 = call i8* @llvm.wasm.get.exception(token %6)
  %8 = call i32 @llvm.wasm.get.ehselector(token %6)
  %9 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %8, %9
  br i1 %matches, label %catch4, label %rethrow
; CHECK: catch.start3:
; CHECK:   call void @llvm.wasm.landingpad.index(token %{{.+}}, i32 0)

catch4:                                           ; preds = %catch.start3
  %10 = call i8* @__cxa_begin_catch(i8* %7) [ "funclet"(token %6) ]
  call void @__cxa_end_catch() [ "funclet"(token %6) ]
  catchret from %6 to label %try.cont7

rethrow:                                          ; preds = %catch.start3
  call void @llvm.wasm.rethrow() [ "funclet"(token %6) ]
  unreachable

try.cont7:                                        ; preds = %try.cont, %catch4
  ret void
}

; A nested try-catch within a catch. The outer catch catches 'int'.
;
; void foo();
; void test2() {
;   try {
;     foo();
;   } catch (int) {
;     try {
;       foo();
;     } catch (int) {
;     }
;   }
; }
; Within the nested catchpad, wasm.lsda() call should NOT be generated.
define void @test2() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test2()
entry:
  invoke void @foo()
          to label %try.cont9 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %rethrow
; CHECK: catch.start:
; CHECK:   call i8* @llvm.wasm.lsda()

catch:                                            ; preds = %catch.start
  %5 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  invoke void @foo() [ "funclet"(token %1) ]
          to label %try.cont unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %catch
  %6 = catchswitch within %1 [label %catch.start3] unwind label %ehcleanup

catch.start3:                                     ; preds = %catch.dispatch2
  %7 = catchpad within %6 [i8* bitcast (i8** @_ZTIi to i8*)]
  %8 = call i8* @llvm.wasm.get.exception(token %7)
  %9 = call i32 @llvm.wasm.get.ehselector(token %7)
  %10 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches4 = icmp eq i32 %9, %10
  br i1 %matches4, label %catch6, label %rethrow5
; CHECK: catch.start3:
; CHECK-NOT:   call i8* @llvm.wasm.lsda()

catch6:                                           ; preds = %catch.start3
  %11 = call i8* @__cxa_begin_catch(i8* %8) [ "funclet"(token %7) ]
  call void @__cxa_end_catch() [ "funclet"(token %7) ]
  catchret from %7 to label %try.cont

rethrow5:                                         ; preds = %catch.start3
  invoke void @llvm.wasm.rethrow() [ "funclet"(token %7) ]
          to label %unreachable unwind label %ehcleanup

try.cont:                                         ; preds = %catch, %catch6
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont9

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont9:                                        ; preds = %entry, %try.cont
  ret void

ehcleanup:                                        ; preds = %rethrow5, %catch.dispatch2
  %12 = cleanuppad within %1 []
  call void @__cxa_end_catch() [ "funclet"(token %12) ]
  cleanupret from %12 unwind to caller
; CHECK: ehcleanup:
; CHECK-NEXT:   cleanuppad
; CHECK-NOT:   call void @llvm.wasm.landingpad.index
; CHECK-NOT:   store {{.*}} @__wasm_lpad_context
; CHECK-NOT:   call i8* @llvm.wasm.lsda()
; CHECK-NOT:   call i32 @_Unwind_CallPersonality
; CHECK-NOT:   load {{.*}} @__wasm_lpad_context

unreachable:                                      ; preds = %rethrow5
  unreachable
}

; A nested try-catch within a catch. The outer catch is (...).
;
; void foo();
; void test2() {
;   try {
;     foo();
;   } catch (...) {
;     try {
;       foo();
;     } catch (int) {
;     }
;   }
; }
; Within the innermost catchpad, wasm.lsda() call should be generated, because
; the outer catch is 'catch (...)', which does not need wasm.lsda() call.
define void @test3() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test3()
entry:
  invoke void @foo()
          to label %try.cont8 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  invoke void @foo() [ "funclet"(token %1) ]
          to label %try.cont unwind label %catch.dispatch2
; CHECK: catch.start:
; CHECK-NOT:   call i8* @llvm.wasm.lsda()

catch.dispatch2:                                  ; preds = %catch.start
  %5 = catchswitch within %1 [label %catch.start3] unwind label %ehcleanup

catch.start3:                                     ; preds = %catch.dispatch2
  %6 = catchpad within %5 [i8* bitcast (i8** @_ZTIi to i8*)]
  %7 = call i8* @llvm.wasm.get.exception(token %6)
  %8 = call i32 @llvm.wasm.get.ehselector(token %6)
  %9 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %8, %9
  br i1 %matches, label %catch4, label %rethrow
; CHECK: catch.start3:
; CHECK:   call i8* @llvm.wasm.lsda()

catch4:                                           ; preds = %catch.start3
  %10 = call i8* @__cxa_begin_catch(i8* %7) [ "funclet"(token %6) ]
  %11 = bitcast i8* %10 to i32*
  %12 = load i32, i32* %11, align 4
  call void @__cxa_end_catch() [ "funclet"(token %6) ]
  catchret from %6 to label %try.cont

rethrow:                                          ; preds = %catch.start3
  invoke void @llvm.wasm.rethrow() [ "funclet"(token %6) ]
          to label %unreachable unwind label %ehcleanup

try.cont:                                         ; preds = %catch.start, %catch4
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont8

try.cont8:                                        ; preds = %entry, %try.cont
  ret void

ehcleanup:                                        ; preds = %rethrow, %catch.dispatch2
  %13 = cleanuppad within %1 []
  invoke void @__cxa_end_catch() [ "funclet"(token %13) ]
          to label %invoke.cont6 unwind label %terminate

invoke.cont6:                                     ; preds = %ehcleanup
  cleanupret from %13 unwind to caller

unreachable:                                      ; preds = %rethrow
  unreachable

terminate:                                        ; preds = %ehcleanup
  %14 = cleanuppad within %13 []
  %15 = call i8* @llvm.wasm.get.exception(token %14)
  call void @__clang_call_terminate(i8* %15) [ "funclet"(token %14) ]
  unreachable
}

; void foo();
; void test4() {
;   try {
;     foo();
;   } catch (int) {
;     try {
;       foo();
;     } catch (...) {
;       try {
;         foo();
;       } catch (int) {
;       }
;     }
;   }
; }
; wasm.lsda() call should be generated only once in the outermost catchpad. The
; innermost 'catch (int)' can reuse the wasm.lsda() generated in the outermost
; catch.
define void @test4() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test4()
entry:
  invoke void @foo()
          to label %try.cont19 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*)]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %rethrow
; CHECK: catch.start:
; CHECK:   call i8* @llvm.wasm.lsda()

catch:                                            ; preds = %catch.start
  %5 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  invoke void @foo() [ "funclet"(token %1) ]
          to label %try.cont16 unwind label %catch.dispatch2

catch.dispatch2:                                  ; preds = %catch
  %8 = catchswitch within %1 [label %catch.start3] unwind label %ehcleanup17

catch.start3:                                     ; preds = %catch.dispatch2
  %9 = catchpad within %8 [i8* null]
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  %11 = call i32 @llvm.wasm.get.ehselector(token %9)
  %12 = call i8* @__cxa_begin_catch(i8* %10) [ "funclet"(token %9) ]
  invoke void @foo() [ "funclet"(token %9) ]
          to label %try.cont unwind label %catch.dispatch7
; CHECK: catch.start3:
; CHECK-NOT:   call i8* @llvm.wasm.lsda()

catch.dispatch7:                                  ; preds = %catch.start3
  %13 = catchswitch within %9 [label %catch.start8] unwind label %ehcleanup

catch.start8:                                     ; preds = %catch.dispatch7
  %14 = catchpad within %13 [i8* bitcast (i8** @_ZTIi to i8*)]
  %15 = call i8* @llvm.wasm.get.exception(token %14)
  %16 = call i32 @llvm.wasm.get.ehselector(token %14)
  %17 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches9 = icmp eq i32 %16, %17
  br i1 %matches9, label %catch11, label %rethrow10
; CHECK: catch.start8:
; CHECK-NOT:   call i8* @llvm.wasm.lsda()

catch11:                                          ; preds = %catch.start8
  %18 = call i8* @__cxa_begin_catch(i8* %15) [ "funclet"(token %14) ]
  %19 = bitcast i8* %18 to i32*
  %20 = load i32, i32* %19, align 4
  call void @__cxa_end_catch() [ "funclet"(token %14) ]
  catchret from %14 to label %try.cont

rethrow10:                                        ; preds = %catch.start8
  invoke void @llvm.wasm.rethrow() [ "funclet"(token %14) ]
          to label %unreachable unwind label %ehcleanup

try.cont:                                         ; preds = %catch.start3, %catch11
  invoke void @__cxa_end_catch() [ "funclet"(token %9) ]
          to label %invoke.cont13 unwind label %ehcleanup17

invoke.cont13:                                    ; preds = %try.cont
  catchret from %9 to label %try.cont16

try.cont16:                                       ; preds = %catch, %invoke.cont13
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont19

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow() [ "funclet"(token %1) ]
  unreachable

try.cont19:                                       ; preds = %entry, %try.cont16
  ret void

ehcleanup:                                        ; preds = %rethrow10, %catch.dispatch7
  %21 = cleanuppad within %9 []
  invoke void @__cxa_end_catch() [ "funclet"(token %21) ]
          to label %invoke.cont14 unwind label %terminate

invoke.cont14:                                    ; preds = %ehcleanup
  cleanupret from %21 unwind label %ehcleanup17

ehcleanup17:                                      ; preds = %invoke.cont14, %try.cont, %catch.dispatch2
  %22 = cleanuppad within %1 []
  call void @__cxa_end_catch() [ "funclet"(token %22) ]
  cleanupret from %22 unwind to caller

unreachable:                                      ; preds = %rethrow10
  unreachable

terminate:                                        ; preds = %ehcleanup
  %23 = cleanuppad within %21 []
  %24 = call i8* @llvm.wasm.get.exception(token %23)
  call void @__clang_call_terminate(i8* %24) [ "funclet"(token %23) ]
  unreachable
}

; A cleanuppad with a call to __clang_call_terminate().
;
; void foo();
; void test5() {
;   try {
;     foo();
;   } catch (...) {
;     foo();
;   }
; }
define void @test5() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test5
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
; CHECK: terminate:
; CHECK-NEXT: cleanuppad
; CHECK-NEXT:   %[[EXN:.*]] = call i8* @llvm.wasm.catch
; CHECK-NEXT:   call void @__clang_call_terminate(i8* %[[EXN]])
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
; void test6() {
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
define void @test6() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @test6
entry:
  %t = alloca %struct.Temp, align 1
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @foo()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %invoke.cont
  %call = call %struct.Temp* @_ZN4TempD2Ev(%struct.Temp* %t)
  br label %try.cont

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %num.0 = phi i32 [ 2, %invoke.cont ], [ 1, %entry ]
  %0 = cleanuppad within none []
  %call2 = call %struct.Temp* @_ZN4TempD2Ev(%struct.Temp* %t) [ "funclet"(token %0) ]
  cleanupret from %0 unwind label %catch.dispatch
; CHECK: ehcleanup:
; CHECK-NEXT:   = phi

catch.dispatch:                                   ; preds = %ehcleanup
  %1 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* null]
  %3 = call i8* @llvm.wasm.get.exception(token %2)
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  %5 = call i8* @__cxa_begin_catch(i8* %3) [ "funclet"(token %2) ]
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
  %7 = catchpad within %6 [i8* null]
  %8 = call i8* @llvm.wasm.get.exception(token %7)
  %9 = call i32 @llvm.wasm.get.ehselector(token %7)
  %10 = call i8* @__cxa_begin_catch(i8* %8) [ "funclet"(token %7) ]
  call void @bar(i32 %num.1) [ "funclet"(token %7) ]
  call void @__cxa_end_catch() [ "funclet"(token %7) ]
  catchret from %7 to label %try.cont10

try.cont10:                                       ; preds = %invoke.cont3, %catch.start6
  ret void
}

; Tests if instructions after a call to @llvm.wasm.throw are deleted and the
; BB's dead children are deleted.

; CHECK-LABEL: @test7
define i32 @test7(i1 %b, i8* %p) {
entry:
  br i1 %b, label %bb.true, label %bb.false

; CHECK:      bb.true:
; CHECK-NEXT:   call void @llvm.wasm.throw(i32 0, i8* %p)
; CHECK-NEXT:   unreachable
bb.true:                                          ; preds = %entry
  call void @llvm.wasm.throw(i32 0, i8* %p)
  br label %bb.true.0

; CHECK-NOT:  bb.true.0
bb.true.0:                                        ; preds = %bb.true
  br label %merge

; CHECK:      bb.false
bb.false:                                         ; preds = %entry
  br label %merge

; CHECK:      merge
merge:                                            ; preds = %bb.true.0, %bb.false
  ret i32 0
}

declare void @foo()
declare void @bar(i32)
declare %struct.Temp* @_ZN4TempD2Ev(%struct.Temp* returned)
declare i32 @__gxx_wasm_personality_v0(...)
declare i8* @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare i32 @llvm.eh.typeid.for(i8*)
declare void @llvm.wasm.throw(i32, i8*)
declare void @llvm.wasm.rethrow()
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__clang_call_terminate(i8*)

; CHECK-DAG: declare void @llvm.wasm.landingpad.index(token, i32 immarg)
; CHECK-DAG: declare i8* @llvm.wasm.lsda()
; CHECK-DAG: declare i32 @_Unwind_CallPersonality(i8*)
