; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128-ni:1"
target triple = "wasm32-unknown-unknown"

%struct.quux = type { i32 }
%struct.blam = type <{ %struct.quux }>

declare void @foo()
declare void @bar(%struct.quux*)
declare i32 @__gxx_wasm_personality_v0(...)

define void @test() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb:
  %tmp0 = alloca %struct.blam, align 4
  br i1 undef, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  %tmp1 = getelementptr inbounds %struct.blam, %struct.blam* %tmp0, i32 0, i32 0
  invoke void @foo()
          to label %bb3 unwind label %bb4

bb2:                                              ; preds = %bb
  %tmp2 = getelementptr inbounds %struct.blam, %struct.blam* %tmp0, i32 0, i32 0
  invoke void @foo()
          to label %bb3 unwind label %bb4

bb3:                                              ; preds = %bb2, %bb1
  unreachable

bb4:                                              ; preds = %bb2, %bb1
  ; This PHI should not be combined into a non-PHI instruction, because
  ; catchswitch BB cannot have any non-PHI instruction other than catchswitch
  ; itself.
  ; CHECK: bb4:
  ; CHECK-NEXT: phi
  ; CHECK-NEXT: catchswitch
  %tmp3 = phi %struct.quux* [ %tmp1, %bb1 ], [ %tmp2, %bb2 ]
  %tmp4 = catchswitch within none [label %bb5] unwind label %bb7

bb5:                                              ; preds = %bb4
  %tmp5 = catchpad within %tmp4 [i8* null]
  invoke void @foo() [ "funclet"(token %tmp5) ]
          to label %bb6 unwind label %bb7

bb6:                                              ; preds = %bb5
  unreachable

bb7:                                              ; preds = %bb5, %bb4
  %tmp6 = cleanuppad within none []
  call void @bar(%struct.quux* %tmp3) [ "funclet"(token %tmp6) ]
  unreachable
}
