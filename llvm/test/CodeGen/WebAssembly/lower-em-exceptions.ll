; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions -S | FileCheck %s -DPTR=i32
; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions -S --mattr=+atomics,+bulk-memory | FileCheck %s -DPTR=i32
; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions --mtriple=wasm64-unknown-unknown -data-layout="e-m:e-p:64:64-i64:64-n32:64-S128" -S | FileCheck %s -DPTR=i64

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@_ZTIi = external constant ptr
@_ZTIc = external constant ptr
; CHECK: @__THREW__ = external thread_local global [[PTR]]
; __threwValue is only used in Emscripten SjLj, so it shouldn't be generated.
; CHECK-NOT: @__threwValue =

; Test invoke instruction with clauses (try-catch block)
define void @clause() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @clause(
entry:
  invoke void @foo(i32 3)
          to label %invoke.cont unwind label %lpad
; CHECK: entry:
; CHECK-NEXT: store [[PTR]] 0, ptr @__THREW__
; CHECK-NEXT: call cc{{.*}} void @__invoke_void_i32(ptr @foo, i32 3)
; CHECK-NEXT: %[[__THREW__VAL:.*]] = load [[PTR]], ptr @__THREW__
; CHECK-NEXT: store [[PTR]] 0, ptr @__THREW__
; CHECK-NEXT: %cmp = icmp eq [[PTR]] %[[__THREW__VAL]], 1
; CHECK-NEXT: br i1 %cmp, label %lpad, label %invoke.cont

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %catch.dispatch
; CHECK: lpad:
; CHECK-NEXT: %[[FMC:.*]] = call ptr @__cxa_find_matching_catch_4(ptr @_ZTIi, ptr null)
; CHECK-NEXT: %[[IVI1:.*]] = insertvalue { ptr, i32 } poison, ptr %[[FMC]], 0
; CHECK-NEXT: %[[TEMPRET0_VAL:.*]] = call i32 @getTempRet0()
; CHECK-NEXT: %[[IVI2:.*]] = insertvalue { ptr, i32 } %[[IVI1]], i32 %[[TEMPRET0_VAL]], 1
; CHECK-NEXT: extractvalue { ptr, i32 } %[[IVI2]], 0
; CHECK-NEXT: %[[CDR:.*]] = extractvalue { ptr, i32 } %[[IVI2]], 1

catch.dispatch:                                   ; preds = %lpad
  %3 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch1, label %catch
; CHECK: catch.dispatch:
; CHECK-NEXT: %[[TYPEID:.*]] = call i32 @llvm_eh_typeid_for(ptr @_ZTIi)
; CHECK-NEXT: %matches = icmp eq i32 %[[CDR]], %[[TYPEID]]

catch1:                                           ; preds = %catch.dispatch
  %4 = call ptr @__cxa_begin_catch(ptr %1)
  %5 = load i32, ptr %4, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %catch1, %invoke.cont
  ret void

catch:                                            ; preds = %catch.dispatch
  %6 = call ptr @__cxa_begin_catch(ptr %1)
  call void @__cxa_end_catch()
  br label %try.cont
}

; Test invoke instruction with filters (functions with throw(...) declaration)
; Currently we don't support exception specifications correctly in JS glue code,
; so we ignore all filters here.
; See https://github.com/llvm/llvm-project/issues/49740.
define void @filter() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @filter(
entry:
  invoke void @foo(i32 3)
          to label %invoke.cont unwind label %lpad
; CHECK: entry:
; CHECK-NEXT: store [[PTR]] 0, ptr @__THREW__
; CHECK-NEXT: call cc{{.*}} void @__invoke_void_i32(ptr @foo, i32 3)
; CHECK-NEXT: %[[__THREW__VAL:.*]] = load [[PTR]], ptr @__THREW__
; CHECK-NEXT: store [[PTR]] 0, ptr @__THREW__
; CHECK-NEXT: %cmp = icmp eq [[PTR]] %[[__THREW__VAL]], 1
; CHECK-NEXT: br i1 %cmp, label %lpad, label %invoke.cont

invoke.cont:                                      ; preds = %entry
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          filter [2 x ptr] [ptr @_ZTIi, ptr @_ZTIc]
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %filter.dispatch
; CHECK: lpad:
; We now temporarily ignore filters because of the bug, so we pass nothing to
; __cxa_find_matching_catch
; CHECK-NEXT: %[[FMC:.*]] = call ptr @__cxa_find_matching_catch_2()

filter.dispatch:                                  ; preds = %lpad
  %ehspec.fails = icmp slt i32 %2, 0
  br i1 %ehspec.fails, label %ehspec.unexpected, label %eh.resume

ehspec.unexpected:                                ; preds = %filter.dispatch
  call void @__cxa_call_unexpected(ptr %1) #4
  unreachable

eh.resume:                                        ; preds = %filter.dispatch
  %lpad.val = insertvalue { ptr, i32 } poison, ptr %1, 0
  %lpad.val3 = insertvalue { ptr, i32 } %lpad.val, i32 %2, 1
  resume { ptr, i32 } %lpad.val3
; CHECK: eh.resume:
; CHECK-NEXT: insertvalue
; CHECK-NEXT: %[[LPAD_VAL:.*]] = insertvalue
; CHECK-NEXT: %[[LOW:.*]] = extractvalue { ptr, i32 } %[[LPAD_VAL]], 0
; CHECK-NEXT: call void @__resumeException(ptr %[[LOW]])
; CHECK-NEXT: unreachable
}

; Test if argument attributes indices in newly created call instructions are correct
define void @arg_attributes() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @arg_attributes(
entry:
  %0 = invoke noalias ptr @bar(i8 signext 1, i8 zeroext 2)
          to label %invoke.cont unwind label %lpad
; CHECK: entry:
; CHECK-NEXT: store [[PTR]] 0, ptr @__THREW__
; CHECK-NEXT: %0 = call cc{{.*}} noalias ptr @__invoke_ptr_i8_i8(ptr @bar, i8 signext 1, i8 zeroext 2)

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %1 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr null
  %2 = extractvalue { ptr, i32 } %1, 0
  %3 = extractvalue { ptr, i32 } %1, 1
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %4 = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch1, label %catch

catch1:                                           ; preds = %catch.dispatch
  %5 = call ptr @__cxa_begin_catch(ptr %2)
  %6 = load i32, ptr %5, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %catch1, %invoke.cont
  ret void

catch:                                            ; preds = %catch.dispatch
  %7 = call ptr @__cxa_begin_catch(ptr %2)
  call void @__cxa_end_catch()
  br label %try.cont
}

declare void @foo(i32)
declare ptr @bar(i8, i8)

declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for.p0(ptr)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @__cxa_call_unexpected(ptr)

; JS glue functions and invoke wrappers declaration
; CHECK-DAG: declare i32 @getTempRet0()
; CHECK-DAG: declare void @__resumeException(ptr)
; CHECK-DAG: declare void @__invoke_void_i32(ptr, i32)
; CHECK-DAG: declare ptr @__cxa_find_matching_catch_4(ptr, ptr)
