; REQUIRES: asserts
; RUN: llc < %s -O0 -frame-pointer=all -relocation-model=pic -stats 2>&1 | FileCheck %s
;
; This test should not cause any spilling with RAFast.
;
; CHECK: Number of copies coalesced
; CHECK-NOT: Number of stores added
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%0 = type { i64, i64, i8*, i8* }
%1 = type opaque
%2 = type opaque
%3 = type <{ i8*, i32, i32, void (%4*)*, i8*, i64 }>
%4 = type { i8**, i32, i32, i8**, %5*, i64 }
%5 = type { i64, i64 }
%6 = type { i8*, i32, i32, i8*, %5* }

@0 = external hidden constant %0

define hidden void @f() ssp {
bb:
  %tmp5 = alloca i64, align 8
  %tmp6 = alloca void ()*, align 8
  %tmp7 = alloca %3, align 8
  store i64 0, i64* %tmp5, align 8
  br label %bb8

bb8:                                              ; preds = %bb23, %bb
  %tmp15 = getelementptr inbounds %3, %3* %tmp7, i32 0, i32 4
  store i8* bitcast (%0* @0 to i8*), i8** %tmp15
  %tmp16 = bitcast %3* %tmp7 to void ()*
  store void ()* %tmp16, void ()** %tmp6, align 8
  %tmp17 = load void ()*, void ()** %tmp6, align 8
  %tmp18 = bitcast void ()* %tmp17 to %6*
  %tmp19 = getelementptr inbounds %6, %6* %tmp18, i32 0, i32 3
  %tmp20 = bitcast %6* %tmp18 to i8*
  %tmp21 = load i8*, i8** %tmp19
  %tmp22 = bitcast i8* %tmp21 to void (i8*)*
  call void %tmp22(i8* %tmp20)
  br label %bb23

bb23:                                             ; preds = %bb8
  %tmp24 = load i64, i64* %tmp5, align 8
  %tmp25 = add i64 %tmp24, 1
  store i64 %tmp25, i64* %tmp5, align 8
  %tmp26 = icmp ult i64 %tmp25, 10
  br i1 %tmp26, label %bb8, label %bb27

bb27:                                             ; preds = %bb23
  ret void
}
