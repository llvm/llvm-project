; RUN: opt -S -passes=consthoist < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define internal fastcc void @baz(i8* %arg) unnamed_addr personality i8* bitcast (i32 (...)* @wobble to i8*) {
; CHECK-LABEL:  @baz
bb:
  %tmp = invoke noalias dereferenceable(40) i8* @wibble.2(i64 40)
          to label %bb6 unwind label %bb1

bb1:                                              ; preds = %bb
; CHECK: bb1:
; CHECK-NEXT:  %tmp2 = catchswitch within none [label %bb3] unwind label %bb16
  %tmp2 = catchswitch within none [label %bb3] unwind label %bb16

bb3:                                              ; preds = %bb1
  %tmp4 = catchpad within %tmp2 [i8* null, i32 64, i8* null]
  invoke void @spam(i8* null) [ "funclet"(token %tmp4) ]
          to label %bb5 unwind label %bb16

bb5:                                              ; preds = %bb3
  unreachable

bb6:                                              ; preds = %bb
  %tmp7 = icmp eq i8* %arg, null
  br label %bb9


bb9:                                              ; preds = %bb8, %bb6
  %tmp10 = inttoptr i64 -6148914691236517376 to i16*
  %tmp11 = invoke noalias dereferenceable(40) i8* @wibble.2(i64 40)
          to label %bb15 unwind label %bb12

bb12:                                             ; preds = %bb9
  %tmp13 = cleanuppad within none []
  br label %bb14

bb14:                                             ; preds = %bb12
  cleanupret from %tmp13 unwind label %bb16

bb15:                                             ; preds = %bb9
  ret void

bb16:                                             ; preds = %bb14, %bb3, %bb1
  %tmp17 = phi i16* [ inttoptr (i64 -6148914691236517376 to i16*), %bb1 ], [ inttoptr (i64 -6148914691236517376 to i16*), %bb3 ], [ %tmp10, %bb14 ]
  %tmp18 = cleanuppad within none []
  br label %bb19

bb19:                                             ; preds = %bb16
  cleanupret from %tmp18 unwind to caller
}

declare i8* @wibble.2(i64)

declare dso_local void @spam(i8*) local_unnamed_addr

declare i32 @wobble(...)
