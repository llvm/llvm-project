; RUN: opt < %s -passes=globalopt -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

%struct.hashheader = type { i16, i16, i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, [5 x i8], [13 x i8], i8, i8, i8, [228 x i16], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [128 x i8], [100 x [11 x i8]], [100 x i32], [100 x i32], i16 }
%struct.strchartype = type { ptr, ptr, ptr }

@hashheader = internal global %struct.hashheader zeroinitializer, align 32 ; <ptr> [#uses=1]
@chartypes = internal global ptr null ; <ptr> [#uses=1]
; CHECK-NOT: @hashheader
; CHECK-NOT: @chartypes

; based on linit in office-ispell
define void @test() nounwind ssp {
  %1 = load i32, ptr getelementptr inbounds (%struct.hashheader, ptr @hashheader, i64 0, i32 13), align 8 ; <i32> [#uses=1]
  %2 = sext i32 %1 to i64                         ; <i64> [#uses=1]
  %3 = mul i64 %2, ptrtoint (ptr getelementptr (%struct.strchartype, ptr null, i64 1) to i64) ; <i64> [#uses=1]
  %4 = tail call ptr @malloc(i64 %3)              ; <ptr> [#uses=1]
; CHECK-NOT: call ptr @malloc(i64
  store ptr %4, ptr @chartypes, align 8
  ret void
}

declare noalias ptr @malloc(i64) allockind("alloc,uninitialized")
