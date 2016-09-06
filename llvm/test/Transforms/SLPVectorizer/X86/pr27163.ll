; RUN: opt -slp-vectorizer -S < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

%struct.B = type { i64, i64 }

define void @test1(%struct.B* %p) personality i32 (...)* @__CxxFrameHandler3 {
invoke.cont:
  %gep1 = getelementptr inbounds %struct.B, %struct.B* %p, i64 0, i32 0
  %gep2 = getelementptr inbounds %struct.B, %struct.B* %p, i64 0, i32 1
  %load1 = load i64, i64* %gep1, align 8
  %load2 = load i64, i64* %gep2, align 8
  store i64 %load1, i64* %gep1, align 8
  store i64 %load2, i64* %gep2, align 8
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont
  %cs = catchswitch within none [label %invoke.cont1] unwind label %ehcleanup

invoke.cont1:                                     ; preds = %catch.dispatch
  %catch = catchpad within %cs [i8* null, i32 64, i8* null]
  invoke void @throw() [ "funclet"(token %catch) ]
          to label %unreachable unwind label %ehcleanup

ehcleanup:                                        ; preds = %invoke.cont1, %catch.dispatch
  %phi = phi i64 [ %load1, %catch.dispatch ], [ 9, %invoke.cont1 ]
  %cleanup = cleanuppad within none []
  call void @release(i64 %phi) [ "funclet"(token %cleanup) ]
  cleanupret from %cleanup unwind to caller

unreachable:                                      ; preds = %invoke.cont1, %invoke.cont
  unreachable
}


; CHECK-LABEL: define void @test1(
; CHECK: %[[gep:.*]] = getelementptr inbounds %struct.B, %struct.B* %p, i64 0, i32 0
; CHECK: %[[bc:.*]]  = bitcast i64* %[[gep]] to <2 x i64>*
; CHECK: %[[ld:.*]]  = load <2 x i64>, <2 x i64>* %[[bc]], align 8
; CHECK: %[[ee:.*]]  = extractelement <2 x i64> %[[ld]], i32 0

; CHECK: %[[phi:.*]] = phi i64 [ %[[ee]], {{.*}} ], [ 9, {{.*}} ]
; CHECK: call void @release(i64 %[[phi]])

declare i32 @__CxxFrameHandler3(...)

declare void @throw()

declare void @release(i64)
