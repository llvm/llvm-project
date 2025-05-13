; RUN: opt < %s -wasm-lower-em-ehsjlj -wasm-enable-eh -wasm-enable-sjlj -S | FileCheck %s

target triple = "wasm32-unknown-emscripten"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }
@buf = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

; When longjmpable calls are coverted into invokes in Wasm SjLj transformation
; and their unwind destination is an existing catchpad or cleanuppad due to
; maintain the scope structure, the new pred BBs created by invokes and the
; correct incoming values should be added the existing phis in those unwind
; destinations.

; When longjmpable calls are within a cleanuppad.
define void @longjmpable_invoke_phi0() personality ptr @__gxx_wasm_personality_v0 {
; CHECK-LABEL: @longjmpable_invoke_phi0
entry:
  %val.entry = call i32 @llvm.wasm.memory.size.i32(i32 0)
  %0 = call i32 @setjmp(ptr @buf) #2
  invoke void @foo()
          to label %bb1 unwind label %ehcleanup1

bb1:                                              ; preds = %entry
  ; We use llvm.wasm.memory.size intrinsic just to get/use an i32 value. The
  ; reason we use an intrinsic here is to make it not longjmpable. If this can
  ; longjmp, the result will be more complicated and hard to check.
  %val.bb1 = call i32 @llvm.wasm.memory.size.i32(i32 0)
  invoke void @foo()
          to label %bb2 unwind label %ehcleanup0

bb2:                                              ; preds = %bb1
  unreachable

ehcleanup0:                                       ; preds = %bb1
  %1 = cleanuppad within none []
  call void @longjmpable() [ "funclet"(token %1) ]
; CHECK:      ehcleanup0
; CHECK:        invoke void @longjmpable
; CHECK-NEXT:           to label %.noexc unwind label %ehcleanup1
  invoke void @foo() [ "funclet"(token %1) ]
          to label %bb3 unwind label %ehcleanup1

bb3:                                              ; preds = %ehcleanup0
  %val.bb3 = call i32 @llvm.wasm.memory.size.i32(i32 0)
  call void @longjmpable() [ "funclet"(token %1) ]
; CHECK:      bb3:
; CHECK:        invoke void @longjmpable
; CHECK-NEXT:           to label %.noexc1 unwind label %ehcleanup1
  cleanupret from %1 unwind label %ehcleanup1

ehcleanup1:                                       ; preds = %bb3, %ehcleanup0, %entry
  %phi = phi i32 [ %val.entry, %entry ], [ %val.bb1, %ehcleanup0 ], [ %val.bb3, %bb3 ]
; CHECK:      ehcleanup1:
; CHECK-NEXT:   %phi = phi i32 [ %val.entry2, %entry.split.split ], [ %val.bb1, %.noexc ], [ %val.bb3, %.noexc1 ], [ %val.bb1, %ehcleanup0 ], [ %val.bb3, %bb3 ]
  %2 = cleanuppad within none []
  %3 = call i32 @llvm.wasm.memory.size.i32(i32 %phi)
  cleanupret from %2 unwind to caller
}

; When longjmpable calls are within a catchpad.
define void @longjmpable_invoke_phi1() personality ptr @__gxx_wasm_personality_v0 {
; CHECK-LABEL: @longjmpable_invoke_phi1
entry:
  %val.entry = call i32 @llvm.wasm.memory.size.i32(i32 0)
  %0 = call i32 @setjmp(ptr @buf) #2
  invoke void @foo()
          to label %bb1 unwind label %ehcleanup

bb1:                                              ; preds = %entry
  %val.bb1 = call i32 @llvm.wasm.memory.size.i32(i32 0)
  invoke void @foo()
          to label %bb2 unwind label %catch.dispatch

bb2:                                              ; preds = %bb1
  unreachable

catch.dispatch:                                   ; preds = %bb1
  %1 = catchswitch within none [label %catch.start] unwind label %ehcleanup

catch.start:                                      ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr null]
  %3 = call ptr @llvm.wasm.get.exception(token %2)
  %4 = call i32 @llvm.wasm.get.ehselector(token %2)
  call void @longjmpable() [ "funclet"(token %2) ]
; CHECK:      catch.start:
; CHECK:        invoke void @longjmpable
; CHECK-NEXT:           to label %.noexc unwind label %ehcleanup
  invoke void @foo() [ "funclet"(token %2) ]
          to label %bb3 unwind label %ehcleanup

bb3:                                              ; preds = %catch.start
  %val.bb3 = call i32 @llvm.wasm.memory.size.i32(i32 0)
  call void @longjmpable() [ "funclet"(token %2) ]
; CHECK:      bb3:
; CHECK:        invoke void @longjmpable
; CHECK-NEXT:           to label %.noexc1 unwind label %ehcleanup
  invoke void @foo() [ "funclet"(token %2) ]
          to label %bb4 unwind label %ehcleanup

bb4:                                              ; preds = %bb3
  unreachable

ehcleanup:                                        ; preds = %bb3, %catch.start, %catch.dispatch, %entry
  %phi = phi i32 [ %val.entry, %entry ], [ %val.bb1, %catch.dispatch ], [ %val.bb1, %catch.start ], [ %val.bb3, %bb3 ]
; CHECK:      ehcleanup:
; CHECK-NEXT:   %phi = phi i32 [ %val.entry2, %entry.split.split ], [ %val.bb1, %catch.dispatch ], [ %val.bb1, %.noexc ], [ %val.bb3, %.noexc1 ], [ %val.bb1, %catch.start ], [ %val.bb3, %bb3 ]
  %5 = cleanuppad within none []
  %6 = call i32 @llvm.wasm.memory.size.i32(i32 %phi)
  cleanupret from %5 unwind to caller
}

declare i32 @setjmp(ptr)
declare i32 @__gxx_wasm_personality_v0(...)
declare void @foo()
declare void @longjmpable()
declare void @use_i32(i32)
; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.wasm.get.ehselector(token) #0
; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare i32 @llvm.wasm.memory.size.i32(i32) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #2 = { returns_twice }
