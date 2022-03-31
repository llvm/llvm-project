; RUN: opt < %s -passes=sroa -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.foo = type { i32 }
declare i32 @__gxx_wasm_personality_v0(...)
declare void @foo()

; Tests if the SROA pass correctly bails out on rewriting PHIs in a catchswitch
; BB.
; CHECK-LABEL: @test_phi_catchswitch
define void @test_phi_catchswitch() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  ; CHECK: alloca
  %tmp = alloca %struct.foo, align 4
  %tmp2 = getelementptr inbounds %struct.foo, %struct.foo* %tmp, i32 0, i32 0
  invoke void @foo()
          to label %bb3 unwind label %bb10

bb3:                                              ; preds = %entry
  invoke void @foo()
          to label %bb9 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %bb3
  ; While rewriting the alloca in the entry BB, the SROA pass tries to insert a
  ; non-PHI instruction in this BB by calling getFirstInsertionPt(), which is
  ; not possible in a catchswitch BB. This test checks if we correctly bail out
  ; on these cases.
  %tmp5 = phi i32* [ %tmp2, %bb3 ]
  %tmp6 = catchswitch within none [label %catch.start] unwind label %bb10

catch.start:                                      ; preds = %catch.dispatch
  %tmp8 = catchpad within %tmp6 [i8* null]
  unreachable

bb9:                                              ; preds = %bb3
  unreachable

bb10:                                             ; preds = %catch.dispatch, %entry
  %tmp11 = phi i32* [ %tmp2, %entry ], [ %tmp5, %catch.dispatch ]
  %tmp12 = cleanuppad within none []
  store i32 0, i32* %tmp11, align 4
  unreachable
}
