; RUN: opt -S -passes=consthoist < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define internal fastcc void @baz(ptr %arg) unnamed_addr personality ptr @wobble {
; CHECK-LABEL:  @baz
bb:
  %tmp = invoke noalias dereferenceable(40) ptr @wibble.2(i64 40)
          to label %bb6 unwind label %bb1

bb1:                                              ; preds = %bb
; CHECK: bb1:
; CHECK-NEXT:  %tmp2 = catchswitch within none [label %bb3] unwind label %bb16
  %tmp2 = catchswitch within none [label %bb3] unwind label %bb16

bb3:                                              ; preds = %bb1
  %tmp4 = catchpad within %tmp2 [ptr null, i32 64, ptr null]
  invoke void @spam(ptr null) [ "funclet"(token %tmp4) ]
          to label %bb5 unwind label %bb16

bb5:                                              ; preds = %bb3
  unreachable

bb6:                                              ; preds = %bb
  %tmp7 = icmp eq ptr %arg, null
  br label %bb9


bb9:                                              ; preds = %bb8, %bb6
  %tmp10 = inttoptr i64 -6148914691236517376 to ptr
  %tmp11 = invoke noalias dereferenceable(40) ptr @wibble.2(i64 40)
          to label %bb15 unwind label %bb12

bb12:                                             ; preds = %bb9
  %tmp13 = cleanuppad within none []
  br label %bb14

bb14:                                             ; preds = %bb12
  cleanupret from %tmp13 unwind label %bb16

bb15:                                             ; preds = %bb9
  ret void

bb16:                                             ; preds = %bb14, %bb3, %bb1
  %tmp17 = phi ptr [ inttoptr (i64 -6148914691236517376 to ptr), %bb1 ], [ inttoptr (i64 -6148914691236517376 to ptr), %bb3 ], [ %tmp10, %bb14 ]
  %tmp18 = cleanuppad within none []
  br label %bb19

bb19:                                             ; preds = %bb16
  cleanupret from %tmp18 unwind to caller
}

declare ptr @wibble.2(i64)

declare dso_local void @spam(ptr) local_unnamed_addr

declare i32 @wobble(...)
