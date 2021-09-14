; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-cxx-exceptions -enable-emscripten-sjlj -S | FileCheck %s
; RUN: llc < %s -enable-emscripten-cxx-exceptions -enable-emscripten-sjlj -verify-machineinstrs

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; The same test with the same function in lower-em-ehsjlj.ll, both the
; Emscripten EH and Emscripten SjLj are enabled in the same way, and this
; function only does exception handling. The difference is that this module does
; not contain any calls to setjmp or longjmp.
;
; But we still have to check if the thrown value is longjmp and and if so
; rethrow it by calling @emscripten_longjmp, because we link object files using
; wasm-ld, so the module we see in LowerEmscriptenEHSjLj pass is not the whole
; program and there can be a longjmp call within another file.
define void @rethrow_longjmp() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
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
; CHECK-NEXT: %__threwValue.val = load i32, i32* @__threwValue, align 4
; CHECK-NEXT: call void @emscripten_longjmp(i32 %threw.phi, i32 %__threwValue.val
; CHECK-NEXT: unreachable

; CHECK:    tail:
; CHECK-NEXT: %cmp = icmp eq i32 %__THREW__.val, 1
; CHECK-NEXT: br i1 %cmp, label %lpad, label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  %3 = call i8* @__cxa_begin_catch(i8* %1) #5
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %lpad, %entry
 %phi = phi i32 [ undef, %entry ], [ undef, %lpad ]
  ret void
}

declare void @foo()
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @__cxa_throw(i8*, i8*, i8*)
