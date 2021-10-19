; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define i32 @foo(i32* nocapture %A) nounwind uwtable readonly ssp {
vector.ph:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <2 x i32> [ zeroinitializer, %vector.ph ], [ %12, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %A, i64 %index
  %1 = bitcast i32* %0 to <2 x i32>*
  %2 = load <2 x i32>, <2 x i32>* %1, align 4
  %3 = sext <2 x i32> %2 to <2 x i64>
  ;CHECK: cost of 1 {{.*}} extract
  %4 = extractelement <2 x i64> %3, i32 0
  %5 = getelementptr inbounds i32, i32* %A, i64 %4
  ;CHECK: cost of 1 {{.*}} extract
  %6 = extractelement <2 x i64> %3, i32 1
  %7 = getelementptr inbounds i32, i32* %A, i64 %6
  %8 = load i32, i32* %5, align 4
  ;CHECK: cost of 1 {{.*}} insert
  %9 = insertelement <2 x i32> poison, i32 %8, i32 0
  %10 = load i32, i32* %7, align 4
  ;CHECK: cost of 1 {{.*}} insert
  %11 = insertelement <2 x i32> %9, i32 %10, i32 1
  %12 = add nsw <2 x i32> %11, %vec.phi
  %index.next = add i64 %index, 2
  %13 = icmp eq i64 %index.next, 192
  br i1 %13, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  %14 = extractelement <2 x i32> %12, i32 0
  %15 = extractelement <2 x i32> %12, i32 1
  %16 = add i32 %14, %15
  ret i32 %16
}
