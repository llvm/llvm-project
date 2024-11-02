; RUN: opt -passes=newgvn -S -o - < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.with_array = type { [2 x i8], i32, i8 }
%struct.with_vector = type { <2 x i8>, i32, i8 }

@main.obj_with_array = private unnamed_addr constant { [2 x i8], i32, i8, [3 x i8] } { [2 x i8] zeroinitializer, i32 0, i8 1, [3 x i8] undef }, align 4
@array_with_zeroinit = common global %struct.with_array zeroinitializer, align 4

@main.obj_with_vector = private unnamed_addr constant { <2 x i8>, i32, i8, [3 x i8] } { <2 x i8> zeroinitializer, i32 0, i8 1, [3 x i8] undef }, align 4
@vector_with_zeroinit = common global %struct.with_vector zeroinitializer, align 4

define i32 @main() {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 @array_with_zeroinit, ptr align 4 @main.obj_with_array, i64 12, i1 false)
  %0 = load i8, ptr getelementptr inbounds (%struct.with_array, ptr @array_with_zeroinit, i64 0, i32 2), align 4

  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 @vector_with_zeroinit, ptr align 4 @main.obj_with_vector, i64 12, i1 false)
  %1 = load i8, ptr getelementptr inbounds (%struct.with_vector, ptr @vector_with_zeroinit, i64 0, i32 2), align 4
  %conv0 = sext i8 %0 to i32
  %conv1 = sext i8 %1 to i32
  %and = and i32 %conv0, %conv1
  ret i32 %and
; CHECK-LABEL: define i32 @main(
; CHECK: ret i32 1
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)
