; RUN: opt < %s -wasm-lower-em-ehsjlj -wasm-enable-eh -wasm-enable-sjlj -S | FileCheck %s
; RUN: llc < %s -wasm-enable-eh -wasm-enable-sjlj -exception-model=wasm -mattr=+exception-handling -verify-machineinstrs

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }
%struct.Temp = type { i8 }
@_ZL3buf = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

; void test() {
;   int jmpval = setjmp(buf);
;   if (jmpval != 0)
;     return;
;   try {
;     foo();
;   } catch (...) {
;   }
; }
define void @setjmp_and_try() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @setjmp_and_try
entry:
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @_ZL3buf, i32 0, i32 0)) #0
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  invoke void @foo()
  to label %return unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.end
  %0 = catchswitch within none [label %catch.start] unwind to caller
; CHECK:       catch.dispatch:
; CHECK-NEXT:    catchswitch within none [label %catch.start] unwind label %catch.dispatch.longjmp

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) #2 [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %return
; CHECK:       catch.start:
; CHECK:         [[T0:%.*]] = catchpad within {{.*}} [i8* null]
; CHECK:         invoke void @__cxa_end_catch() [ "funclet"(token [[T0]]) ]
; CHECK-NEXT:    to label %.noexc unwind label %catch.dispatch.longjmp

; CHECK:       .noexc:
; CHECK-NEXT:     catchret from [[T0]] to label {{.*}}

return:                                           ; preds = %catch.start, %if.end, %entry
  ret void

; CHECK:       catch.dispatch.longjmp:
; CHECK-NEXT:    catchswitch within none [label %catch.longjmp] unwind to caller
}

; void setjmp_within_try() {
;   try {
;     foo();
;     int jmpval = setjmp(buf);
;     if (jmpval != 0)
;       return;
;     foo();
;   } catch (...) {
;   }
; }
define void @setjmp_within_try() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @setjmp_within_try
entry:
  %jmpval = alloca i32, align 4
  %exn.slot = alloca i8*, align 4
  invoke void @foo()
  to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %entry
  %call = invoke i32 @setjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @_ZL3buf, i32 0, i32 0)) #0
  to label %invoke.cont1 unwind label %catch.dispatch

invoke.cont1:                                     ; preds = %invoke.cont
  store i32 %call, i32* %jmpval, align 4
  %0 = load i32, i32* %jmpval, align 4
  %cmp = icmp ne i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont1
  br label %try.cont

if.end:                                           ; preds = %invoke.cont1
  invoke void @foo()
  to label %invoke.cont2 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.end, %invoke.cont, %entry
  %1 = catchswitch within none [label %catch.start] unwind to caller

; CHECK:       catch.dispatch:
; CHECK:         catchswitch within none [label %catch.start] unwind label %catch.dispatch.longjmp
catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* null]
  %3 = call i8* @llvm.wasm.get.exception(token %2)
  store i8* %3, i8** %exn.slot, align 4
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  br label %catch

catch:                                            ; preds = %catch.start
  %exn = load i8*, i8** %exn.slot, align 4
  %5 = call i8* @__cxa_begin_catch(i8* %exn) #2 [ "funclet"(token %2) ]
  call void @__cxa_end_catch() [ "funclet"(token %2) ]
  catchret from %2 to label %catchret.dest
; CHECK: catch:                                            ; preds = %catch.start
; CHECK-NEXT:   %exn = load i8*, i8** %exn.slot15, align 4
; CHECK-NEXT:   %5 = call i8* @__cxa_begin_catch(i8* %exn) #2 [ "funclet"(token %2) ]
; CHECK-NEXT:   invoke void @__cxa_end_catch() [ "funclet"(token %2) ]
; CHECK-NEXT:           to label %.noexc unwind label %catch.dispatch.longjmp

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %catchret.dest, %if.then
  ret void

invoke.cont2:                                     ; preds = %if.end
  br label %try.cont

; CHECK:       catch.dispatch.longjmp:
; CHECK-NEXT:    catchswitch within none [label %catch.longjmp] unwind to caller
}

; void setjmp_and_nested_try() {
;   int jmpval = setjmp(buf);
;   if (jmpval != 0)
;     return;
;   try {
;     foo();
;     try {
;       foo();
;     } catch (...) {
;       foo();
;     }
;   } catch (...) {
;   }
; }
define void @setjmp_and_nested_try() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @setjmp_and_nested_try
entry:
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @_ZL3buf, i32 0, i32 0)) #0
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %try.cont10, label %if.end

if.end:                                           ; preds = %entry
  invoke void @foo()
  to label %invoke.cont unwind label %catch.dispatch5

invoke.cont:                                      ; preds = %if.end
  invoke void @foo()
  to label %try.cont10 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont
  %0 = catchswitch within none [label %catch.start] unwind label %catch.dispatch5

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) #2 [ "funclet"(token %1) ]
  invoke void @foo() [ "funclet"(token %1) ]
  to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %catch.start
  invoke void @__cxa_end_catch() [ "funclet"(token %1) ]
  to label %invoke.cont3 unwind label %catch.dispatch5

invoke.cont3:                                     ; preds = %invoke.cont2
  catchret from %1 to label %try.cont10

ehcleanup:                                        ; preds = %catch.start
  %5 = cleanuppad within %1 []
  invoke void @__cxa_end_catch() [ "funclet"(token %5) ]
  to label %invoke.cont4 unwind label %terminate
; CHECK:       ehcleanup:
; CHECK-NEXT:    [[T0:%.*]] = cleanuppad within {{.*}} []
; CHECK-NEXT:    invoke void @__cxa_end_catch() [ "funclet"(token [[T0]]) ]
; CHECK-NEXT:    to label %invoke.cont4 unwind label %terminate

invoke.cont4:                                     ; preds = %ehcleanup
  cleanupret from %5 unwind label %catch.dispatch5
; CHECK:       invoke.cont4:
; CHECK-NEXT:    cleanupret from [[T0]] unwind label %catch.dispatch5

catch.dispatch5:                                  ; preds = %invoke.cont4, %invoke.cont2, %catch.dispatch, %if.end
  %6 = catchswitch within none [label %catch.start6] unwind to caller
; CHECK:       catch.dispatch5:
; CHECK-NEXT:    catchswitch within none [label %catch.start6] unwind label %catch.dispatch.longjmp

catch.start6:                                     ; preds = %catch.dispatch5
  %7 = catchpad within %6 [i8* null]
  %8 = call i8* @llvm.wasm.get.exception(token %7)
  %9 = call i32 @llvm.wasm.get.ehselector(token %7)
  %10 = call i8* @__cxa_begin_catch(i8* %8) #2 [ "funclet"(token %7) ]
  call void @__cxa_end_catch() [ "funclet"(token %7) ]
  catchret from %7 to label %try.cont10
; CHECK:       catch.start6:
; CHECK-NEXT:    [[T1:%.*]] = catchpad within {{.*}} [i8* null]
; CHECK-NEXT:    call i8* @llvm.wasm.get.exception(token [[T1]])
; CHECK-NEXT:    call i32 @llvm.wasm.get.ehselector(token [[T1]])
; CHECK-NEXT:    call i8* @__cxa_begin_catch(i8* {{.*}}) {{.*}} [ "funclet"(token [[T1]]) ]
; CHECK:         invoke void @__cxa_end_catch() [ "funclet"(token [[T1]]) ]
; CHECK-NEXT:    to label %.noexc unwind label %catch.dispatch.longjmp

; CHECK:       .noexc:
; CHECK-NEXT:    catchret from [[T1]] to label {{.*}}

try.cont10:                                       ; preds = %catch.start6, %invoke.cont3, %invoke.cont, %entry
  ret void

terminate:                                        ; preds = %ehcleanup
  %11 = cleanuppad within %5 []
  call void @terminate() #3 [ "funclet"(token %11) ]
  unreachable
; CHECK:       terminate:
; CHECK-NEXT:    [[T2:%.*]] = cleanuppad within [[T0]] []
; Note that this call unwinds not to %catch.dispatch.longjmp but to
; %catch.dispatch5. This call is enclosed in the cleanuppad above, but there is
; no matching catchret, which has the unwind destination. So this checks this
; cleanuppad's parent, which is in 'ehcleanup', and unwinds to its unwind
; destination, %catch.dispatch5.
; This call was originally '_ZSt9terminatev', which is the mangled name for
; 'std::terminate'. But we listed that as "cannot longjmp", we changed
; the name of the function in this test to show the case in which a call has to
; change to an invoke whose unwind destination is determined by its parent
; chain.
; CHECK-NEXT:    invoke void @terminate() {{.*}} [ "funclet"(token [[T2]]) ]
; CHECK-NEXT:    to label %.noexc4 unwind label %catch.dispatch5

; CHECK:       .noexc4:
; CHECK-NEXT:    unreachable

; CHECK:       catch.dispatch.longjmp:
; CHECK-NEXT:    catchswitch within none [label %catch.longjmp] unwind to caller
}

; void @cleanuppad_no_parent {
;   jmp_buf buf;
;   Temp t;
;   setjmp(buf);
; }
define void @cleanuppad_no_parent() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @cleanuppad_no_parent
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %t = alloca %struct.Temp, align 1
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = invoke i32 @setjmp(%struct.__jmp_buf_tag* noundef %arraydecay) #0
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %call1 = call noundef %struct.Temp* @_ZN4TempD2Ev(%struct.Temp* noundef %t) #2
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  ; After SjLj transformation, this will be converted to an invoke that
  ; eventually unwinds to %catch.dispatch.longjmp. But in case a call has a
  ; "funclet" attribute, we should unwind to the funclet's unwind destination
  ; first to preserve the scoping structure. But this call's parent is %0
  ; (cleanuppad), whose parent is 'none', so we should unwind directly to
  ; %catch.dispatch.longjmp.
  %call2 = call noundef %struct.Temp* @_ZN4TempD2Ev(%struct.Temp* noundef %t) #2 [ "funclet"(token %0) ]
; CHECK: %call13 = invoke {{.*}} %struct.Temp* @_ZN4TempD2Ev(%struct.Temp*
; CHECK-NEXT:    to label {{.*}} unwind label %catch.dispatch.longjmp
  cleanupret from %0 unwind to caller
}

; This case was adapted from @cleanuppad_no_parent by removing allocas and
; destructor calls, to generate a situation that there's only 'invoke @setjmp'
; and no other longjmpable calls.
define i32 @setjmp_only() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
; CHECK-LABEL: @setjmp_only
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = invoke i32 @setjmp(%struct.__jmp_buf_tag* noundef %arraydecay) #0
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  ret i32 %call
; CHECK: invoke.cont:
; The remaining setjmp call is converted to constant 0, because setjmp returns 0
; when called directly.
; CHECK:   ret i32 0

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  cleanupret from %0 unwind to caller
}

declare void @foo()
; Function Attrs: nounwind
declare %struct.Temp* @_ZN4TempD2Ev(%struct.Temp* %this) #2
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0
; Function Attrs: noreturn
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nounwind
declare i8* @llvm.wasm.get.exception(token) #2
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #2
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @terminate()

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind }
