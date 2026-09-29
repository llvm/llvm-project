; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define i32 @foo(ptr nocapture %A) nounwind uwtable readonly ssp {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <2 x i32> [ zeroinitializer, %vector.ph ], [ %11, %vector.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %index
  %1 = load <2 x i32>, ptr %0, align 4
  %2 = sext <2 x i32> %1 to <2 x i64>
  ;CHECK: cost of 1 {{.*}} extract
  %3 = extractelement <2 x i64> %2, i32 0
  %4 = getelementptr inbounds i32, ptr %A, i64 %3
  ;CHECK: cost of 1 {{.*}} extract
  %5 = extractelement <2 x i64> %2, i32 1
  %6 = getelementptr inbounds i32, ptr %A, i64 %5
  %7 = load i32, ptr %4, align 4
  ;CHECK: cost of 0 {{.*}} insert
  %8 = insertelement <2 x i32> poison, i32 %7, i32 0
  %9 = load i32, ptr %6, align 4
  ;CHECK: cost of 1 {{.*}} insert
  %10 = insertelement <2 x i32> %8, i32 %9, i32 1
  %11 = add nsw <2 x i32> %10, %vec.phi
  %index.next = add i64 %index, 2
  %12 = icmp eq i64 %index.next, 192
  br i1 %12, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  %13 = extractelement <2 x i32> %11, i32 0
  %14 = extractelement <2 x i32> %11, i32 1
  %15 = add i32 %13, %14
  ret i32 %15
}
