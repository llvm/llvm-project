; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128-ni:1"
target triple = "wasm32-unknown-unknown"

%struct.quux = type { i32 }
%struct.blam = type <{ %struct.quux }>

declare void @foo()
declare void @bar(%struct.quux*)
declare i32 @baz()
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: noreturn
declare void @llvm.wasm.rethrow() #0

; Test that a PHI in catchswitch BB are excluded from combining into a non-PHI
; instruction.
define void @test0(i1 %c1) personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
bb:
  %tmp0 = alloca %struct.blam, align 4
  br i1 %c1, label %bb1, label %bb2

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

; Test that slicing-up of illegal integer type PHI does not happen in catchswitch
; BBs, which can't have any non-PHI instruction before the catchswitch.
define void @test1() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @foo()
          to label %invoke.cont unwind label %catch.dispatch1

invoke.cont:                                      ; preds = %entry
  %call = invoke i32 @baz()
          to label %invoke.cont1 unwind label %catch.dispatch

invoke.cont1:                                     ; preds = %invoke.cont
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont1
  br label %if.end

if.end:                                           ; preds = %if.then, %invoke.cont1
  %ap.0 = phi i8 [ 1, %if.then ], [ 0, %invoke.cont1 ]
  invoke void @foo()
          to label %invoke.cont2 unwind label %catch.dispatch

invoke.cont2:                                     ; preds = %if.end
  br label %try.cont

catch.dispatch:                                   ; preds = %if.end, %invoke.cont
  ; %ap.2 in catch.dispatch1 BB has an illegal integer type (i8) in the data
  ; layout, and it is only used by trunc or trunc(lshr) operations. In this case
  ; InstCombine will split this PHI in its predecessors, which include this
  ; catch.dispatch BB. This splitting involves creating non-PHI instructions,
  ; such as 'and' or 'icmp' in this BB, which is not valid for a catchswitch BB.
  ; So if one of sliced-up PHI's predecessor is a catchswitch block, we don't
  ; optimize that case and bail out. This BB should be preserved intact after
  ; InstCombine and the pass shouldn't produce invalid code.
  ; CHECK: catch.dispatch:
  ; CHECK-NEXT: phi
  ; CHECK-NEXT: catchswitch
  %ap.1 = phi i8 [ %ap.0, %if.end ], [ 0, %invoke.cont ]
  %tmp0 = catchswitch within none [label %catch.start] unwind label %catch.dispatch1

catch.start:                                      ; preds = %catch.dispatch
  %tmp1 = catchpad within %tmp0 [i8* null]
  br i1 0, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  catchret from %tmp1 to label %try.cont

rethrow:                                          ; preds = %catch.start
  invoke void @llvm.wasm.rethrow() #0 [ "funclet"(token %tmp1) ]
          to label %unreachable unwind label %catch.dispatch1

catch.dispatch1:                                  ; preds = %rethrow, %catch.dispatch, %entry
  %ap.2 = phi i8 [ %ap.1, %catch.dispatch ], [ %ap.1, %rethrow ], [ 0, %entry ]
  %tmp2 = catchswitch within none [label %catch.start1] unwind to caller

catch.start1:                                     ; preds = %catch.dispatch1
  %tmp3 = catchpad within %tmp2 [i8* null]
  %tobool1 = trunc i8 %ap.2 to i1
  br i1 %tobool1, label %if.then1, label %if.end1

if.then1:                                         ; preds = %catch.start1
  br label %if.end1

if.end1:                                          ; preds = %if.then1, %catch.start1
  catchret from %tmp3 to label %try.cont

try.cont:                                         ; preds = %if.end1, %catch, %invoke.cont2
  ret void

unreachable:                                      ; preds = %rethrow
  unreachable
}

attributes #0 = { noreturn }
