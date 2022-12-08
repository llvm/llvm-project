; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; This is a regression test for a bug in which we used phis() without
; make_early_inc_range() in a for loop while deleting phi nodes.

define void @cleanup_phis() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb0:
  invoke void @foo()
          to label %bb1 unwind label %ehcleanup

bb1:                                              ; preds = %bb0
  invoke void @foo()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %bb1
  ret void

ehcleanup:                                       ; preds = %bb1, %bb0
  %phi0 = phi i32 [ 0, %bb0 ], [ 1, %bb1 ]
  %phi1 = phi i32 [ 2, %bb0 ], [ 3, %bb1 ]
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %catchswitch

; These two phi nodes were originally in ehcleanup. Both phi nodes should be
; correctly copied to this catchswitch BB.
; CHECK: catchswitch:
; CHECK-NEXT:  %phi0 = phi i32 [ 0, %bb0 ], [ 1, %bb1 ]
; CHECK-NEXT:  %phi1 = phi i32 [ 2, %bb0 ], [ 3, %bb1 ]
catchswitch:                                      ; preds = %ehcleanup
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catchswitch
  %2 = catchpad within %1 [i8* null]
  call void @bar(i32 %phi0, i32 %phi1)
  unreachable
}

declare void @foo()
declare void @bar(i32, i32)
declare i32 @__gxx_wasm_personality_v0(...)
