; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions -enable-emscripten-sjlj -S | FileCheck %s
; RUN: llc < %s -enable-emscripten-cxx-exceptions -enable-emscripten-sjlj -verify-machineinstrs

; Tests for cases when exception handling and setjmp/longjmp handling are mixed.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }
@_ZTIi = external constant ptr

; There is a function call (@foo) that can either throw an exception or longjmp
; and there is also a setjmp call. When @foo throws, we have to check both for
; exception and longjmp and jump to exception or longjmp handling BB depending
; on the result.
define void @setjmp_longjmp_exception() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @setjmp_longjmp_exception
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  invoke void @foo()
          to label %try.cont unwind label %lpad

; CHECK:    entry.split.split:
; CHECK:      %[[CMP0:.*]] = icmp ne i32 %__THREW__.val, 0
; CHECK-NEXT: %__threwValue.val = load i32, ptr @__threwValue
; CHECK-NEXT: %[[CMP1:.*]] = icmp ne i32 %__threwValue.val, 0
; CHECK-NEXT: %[[CMP:.*]] = and i1 %[[CMP0]], %[[CMP1]]
; CHECK-NEXT: br i1 %[[CMP]], label %if.then1, label %if.else1

; This is exception checking part. %if.else1 leads here
; CHECK:    entry.split.split.split:
; CHECK-NEXT: %[[CMP:.*]] = icmp eq i32 %__THREW__.val, 1
; CHECK-NEXT: br i1 %[[CMP]], label %lpad, label %try.cont

; CHECK:    lpad:
lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
; CHECK-NOT:  call {{.*}} void @__invoke_void(ptr @__cxa_end_catch)
  %3 = call ptr @__cxa_begin_catch(ptr %1) #2
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %lpad, %entry
  ret void

; longjmp checking part
; CHECK:    if.then1:
; CHECK:      call i32 @__wasm_setjmp_test
}

; @foo can either throw an exception or longjmp. Because this function doesn't
; have any setjmp calls, we only handle exceptions in this function. But because
; sjlj is enabled, we check if the thrown value is longjmp and if so rethrow it
; by calling @emscripten_longjmp.
define void @rethrow_longjmp() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @rethrow_longjmp
entry:
  invoke void @foo()
          to label %try.cont unwind label %lpad
; CHECK:    entry:
; CHECK:      %cmp.eq.one = icmp eq i32 %__THREW__.val, 1
; CHECK-NEXT: %cmp.eq.zero = icmp eq i32 %__THREW__.val, 0
; CHECK-NEXT: %or = or i1 %cmp.eq.zero, %cmp.eq.one
; CHECK-NEXT: br i1 %or, label %tail, label %rethrow.longjmp

; CHECK: try.cont:
; CHECK-NEXT:  %phi = phi i32 [ undef, %tail ], [ undef, %lpad ]
; CHECK-NEXT:  ret void

; CHECK:    rethrow.longjmp:
; CHECK-NEXT: %threw.phi = phi i32 [ %__THREW__.val, %entry ]
; CHECK-NEXT: %__threwValue.val = load i32, ptr @__threwValue, align 4
; CHECK-NEXT: call void @emscripten_longjmp(i32 %threw.phi, i32 %__threwValue.val
; CHECK-NEXT: unreachable

; CHECK:    tail:
; CHECK-NEXT: %cmp = icmp eq i32 %__THREW__.val, 1
; CHECK-NEXT: br i1 %cmp, label %lpad, label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = call ptr @__cxa_begin_catch(ptr %1) #5
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %lpad, %entry
 %phi = phi i32 [ undef, %entry ], [ undef, %lpad ]
  ret void
}

; This function contains a setjmp call and no invoke, so we only handle longjmp
; here. But @foo can also throw an exception, so we check if an exception is
; thrown and if so rethrow it by calling @__resumeException. Also we have to
; free the setjmpTable buffer before calling @__resumeException.
define void @rethrow_exception() {
; CHECK-LABEL: @rethrow_exception
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  call void @foo()
  br label %return

; CHECK:    if.end:
; CHECK:      %cmp.eq.one = icmp eq i32 %__THREW__.val, 1
; CHECK-NEXT: br i1 %cmp.eq.one, label %rethrow.exn, label %normal

; CHECK:    rethrow.exn:
; CHECK-NEXT: %exn = call ptr @__cxa_find_matching_catch_2()
; CHECK-NEXT: call void @__resumeException(ptr %exn)
; CHECK-NEXT: unreachable

; CHECK:    normal:
; CHECK-NEXT: icmp ne i32 %__THREW__.val, 0

return:                                           ; preds = %if.end, %entry
  ret void
}

; The same as 'rethrow_exception' but contains a __cxa_throw call. We have to
; free the setjmpTable buffer before calling __cxa_throw.
define void @rethrow_exception2() {
; CHECK-LABEL: @rethrow_exception2
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %throw, label %if.end

if.end:                                           ; preds = %entry
  call void @foo()
  br label %throw

throw:                                            ; preds = %if.end, %entry
  call void @__cxa_throw(ptr null, ptr null, ptr null) #1
  unreachable

; CHECK:    throw:
; CHECK-NEXT: call void @__cxa_throw(ptr null, ptr null, ptr null)
; CHECK-NEXT: unreachable
}

; The same case with @rethrow_longjmp, but there are multiple function calls
; that can possibly longjmp (instead of throwing exception) so we have to
; rethrow them. Here we test if we correclty generate only one 'rethrow.longjmp'
; BB and share it for multiple calls.
define void @rethrow_longjmp_multi() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @rethrow_longjmp_multi
entry:
  invoke void @foo()
          to label %bb unwind label %lpad

bb:                                               ; preds = %entry
  invoke void @foo()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %bb, %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = call ptr @__cxa_begin_catch(ptr %1) #5
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %lpad, %bb
 %phi = phi i32 [ undef, %bb ], [ undef, %lpad ]
  ret void

; CHECK:    rethrow.longjmp:
; CHECK-NEXT: %threw.phi = phi i32 [ %__THREW__.val, %entry ], [ %__THREW__.val1, %bb ]
; CHECK-NEXT: %__threwValue.val = load i32, ptr @__threwValue, align 4
; CHECK-NEXT: call void @emscripten_longjmp(i32 %threw.phi, i32 %__threwValue.val)
; CHECK-NEXT: unreachable
}

; The same case with @rethrow_exception, but there are multiple function calls
; that can possibly throw (instead of longjmping) so we have to rethrow them.
; Here we test if correctly we generate only one 'rethrow.exn' BB and share it
; for multiple calls.
define void @rethrow_exception_multi() {
; CHECK-LABEL: @rethrow_exception_multi
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  call void @foo()
  call void @foo()
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void

; CHECK:    rethrow.exn:
; CHECK-NEXT: %exn = call ptr @__cxa_find_matching_catch_2()
; CHECK-NEXT: call void @__resumeException(ptr %exn)
; CHECK-NEXT: unreachable
}

; int jmpval = setjmp(buf);
; if (jmpval != 0)
;   return;
; try {
;   throw 3;
; } catch (...) {
; }
define void @setjmp_with_throw_try_catch() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @setjmp_with_throw_try_catch
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  %cmp = icmp ne i32 %call, 0
  br i1 %cmp, label %try.cont, label %if.end

if.end:                                           ; preds = %entry
  %exception = call ptr @__cxa_allocate_exception(i32 4) #2
  store i32 3, ptr %exception, align 16
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null) #1
          to label %unreachable unwind label %lpad
; When invoke @__cxa_throw is converted to a call to the invoke wrapper,
; "noreturn" attribute should be removed, and there should be no 'free' call
; before the call. We insert a 'free' call that frees 'setjmpTable' before every
; function-exiting instruction. And invoke wrapper calls shouldn't be treated as
; noreturn instructions, because they are supposed to return.
; CHECK:   if.end:
; CHECK-NOT: tail call void @free
; CHECK-NOT: call cc99 void @__invoke_void_ptr_ptr_ptr(ptr @__cxa_throw, ptr %exception, ptr @_ZTIi, ptr null) #
; CHECK:     call cc99 void @__invoke_void_ptr_ptr_ptr(ptr @__cxa_throw, ptr %exception, ptr @_ZTIi, ptr null)

lpad:                                             ; preds = %if.end
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = call ptr @__cxa_begin_catch(ptr %1) #2
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  ret void

unreachable:                                      ; preds = %if.end
  unreachable
}

declare void @foo()
; Function Attrs: returns_twice
declare i32 @setjmp(ptr)
; Function Attrs: noreturn
declare void @longjmp(ptr, i32)
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @__cxa_throw(ptr, ptr, ptr)
declare ptr @__cxa_allocate_exception(i32)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
