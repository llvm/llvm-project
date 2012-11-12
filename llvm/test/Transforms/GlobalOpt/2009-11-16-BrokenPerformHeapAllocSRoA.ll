; RUN: opt < %s -globalopt -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

%struct.hashheader = type { i16, i16, i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, [5 x i8], [13 x i8], i8, i8, i8, [228 x i16], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [128 x i8], [100 x [11 x i8]], [100 x i32], [100 x i32], i16 }
%struct.strchartype = type { i8*, i8*, i8* }

@hashheader = internal global %struct.hashheader zeroinitializer, align 32 ; <%struct.hashheader*> [#uses=1]
@chartypes = internal global %struct.strchartype* null ; <%struct.strchartype**> [#uses=1]
; CHECK-NOT: @hashheader
; CHECK-NOT: @chartypes

; based on linit in office-ispell
define void @test() nounwind ssp {
  %1 = load i32* getelementptr inbounds (%struct.hashheader* @hashheader, i64 0, i32 13), align 8 ; <i32> [#uses=1]
  %2 = sext i32 %1 to i64                         ; <i64> [#uses=1]
  %3 = mul i64 %2, ptrtoint (%struct.strchartype* getelementptr (%struct.strchartype* null, i64 1) to i64) ; <i64> [#uses=1]
  %4 = tail call i8* @malloc(i64 %3)              ; <i8*> [#uses=1]
; CHECK-NOT: call i8* @malloc(i64
  %5 = bitcast i8* %4 to %struct.strchartype*     ; <%struct.strchartype*> [#uses=1]
  store %struct.strchartype* %5, %struct.strchartype** @chartypes, align 8
  ret void
}

declare noalias i8* @malloc(i64)
