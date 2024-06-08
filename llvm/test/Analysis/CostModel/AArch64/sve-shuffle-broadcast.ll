; Check getShuffleCost for SK_BroadCast with scalable vector

; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=sve  < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void  @broadcast() #0{
; CHECK-LABEL: 'broadcast'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %zero = shufflevector <vscale x 16 x i8> undef, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
; CHECK-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %1 = shufflevector <vscale x 32 x i8> undef, <vscale x 32 x i8> undef, <vscale x 32 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %2 = shufflevector <vscale x 8 x i16> undef, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %3 = shufflevector <vscale x 16 x i16> undef, <vscale x 16 x i16> undef, <vscale x 16 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %4 = shufflevector <vscale x 4 x i32> undef, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %5 = shufflevector <vscale x 8 x i32> undef, <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %6 = shufflevector <vscale x 2 x i64> undef, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %7 = shufflevector <vscale x 4 x i64> undef, <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %8 = shufflevector <vscale x 2 x half> undef, <vscale x 2 x half> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %9 = shufflevector <vscale x 4 x half> undef, <vscale x 4 x half> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %10 = shufflevector <vscale x 8 x half> undef, <vscale x 8 x half> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %11 = shufflevector <vscale x 16 x half> undef, <vscale x 16 x half> undef, <vscale x 16 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %12 = shufflevector <vscale x 2 x float> undef, <vscale x 2 x float> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %13 = shufflevector <vscale x 4 x float> undef, <vscale x 4 x float> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %14 = shufflevector <vscale x 8 x float> undef, <vscale x 8 x float> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %15 = shufflevector <vscale x 2 x double> undef, <vscale x 2 x double> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %16 = shufflevector <vscale x 4 x double> undef, <vscale x 4 x double> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %17 = shufflevector <vscale x 2 x bfloat> undef, <vscale x 2 x bfloat> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %18 = shufflevector <vscale x 4 x bfloat> undef, <vscale x 4 x bfloat> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %19 = shufflevector <vscale x 8 x bfloat> undef, <vscale x 8 x bfloat> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %20 = shufflevector <vscale x 16 x bfloat> undef, <vscale x 16 x bfloat> undef, <vscale x 16 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %21 = shufflevector <vscale x 16 x i1> undef, <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %22 = shufflevector <vscale x 8 x i1> undef, <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %23 = shufflevector <vscale x 4 x i1> undef, <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %24 = shufflevector <vscale x 2 x i1> undef, <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret void

  %zero = shufflevector <vscale x 16 x i8> undef, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %1 = shufflevector <vscale x 32 x i8> undef, <vscale x 32 x i8> undef, <vscale x 32 x i32> zeroinitializer
  %2 = shufflevector <vscale x 8 x i16> undef, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %3 = shufflevector <vscale x 16 x i16> undef, <vscale x 16 x i16> undef, <vscale x 16 x i32> zeroinitializer
  %4 = shufflevector <vscale x 4 x i32> undef, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %5 = shufflevector <vscale x 8 x i32> undef, <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
  %6 = shufflevector <vscale x 2 x i64> undef, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %7 = shufflevector <vscale x 4 x i64> undef, <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer
  %8 = shufflevector <vscale x 2 x half> undef, <vscale x 2 x half> undef, <vscale x 2 x i32> zeroinitializer
  %9 = shufflevector <vscale x 4 x half> undef, <vscale x 4 x half> undef, <vscale x 4 x i32> zeroinitializer
  %10 = shufflevector <vscale x 8 x half> undef, <vscale x 8 x half> undef, <vscale x 8 x i32> zeroinitializer
  %11 = shufflevector <vscale x 16 x half> undef, <vscale x 16 x half> undef, <vscale x 16 x i32> zeroinitializer
 %12 = shufflevector <vscale x 2 x float> undef, <vscale x 2 x float> undef, <vscale x 2 x i32> zeroinitializer
 %13 = shufflevector <vscale x 4 x float> undef, <vscale x 4 x float> undef, <vscale x 4 x i32> zeroinitializer
 %14 = shufflevector <vscale x 8 x float> undef, <vscale x 8 x float> undef, <vscale x 8 x i32> zeroinitializer
 %15 = shufflevector <vscale x 2 x double> undef, <vscale x 2 x double> undef, <vscale x 2 x i32> zeroinitializer
 %16 = shufflevector <vscale x 4 x double> undef, <vscale x 4 x double> undef, <vscale x 4 x i32> zeroinitializer
 %17 = shufflevector <vscale x 2 x bfloat> undef, <vscale x 2 x bfloat> undef, <vscale x 2 x i32> zeroinitializer
 %18 = shufflevector <vscale x 4 x bfloat> undef, <vscale x 4 x bfloat> undef, <vscale x 4 x i32> zeroinitializer
 %19 = shufflevector <vscale x 8 x bfloat> undef, <vscale x 8 x bfloat> undef, <vscale x 8 x i32> zeroinitializer
 %20 = shufflevector <vscale x 16 x bfloat> undef, <vscale x 16 x bfloat> undef, <vscale x 16 x i32> zeroinitializer
 %21 = shufflevector <vscale x 16 x i1> undef, <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
 %22 = shufflevector <vscale x 8 x i1> undef, <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
 %23 = shufflevector <vscale x 4 x i1> undef, <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
 %24 = shufflevector <vscale x 2 x i1> undef, <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
  ret void
}

attributes #0 = { "target-features"="+sve,+bf16" }
