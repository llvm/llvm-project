; RUN: llc < %s -mtriple=x86_64 -mattr=+avx2 -o /dev/null
;
; Verify that CSE-reusing a TokenFactor during broadcast-load combining does not
; introduce a self-reference while replacing the old chain.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f18() #0 {
entry:
  %0 = load <15 x i64>, ptr null, align 128
  %shuffle = shufflevector <15 x i64> %0, <15 x i64> zeroinitializer, <16 x i32> <i32 7, i32 6, i32 4, i32 12, i32 9, i32 5, i32 3, i32 5, i32 12, i32 13, i32 0, i32 14, i32 7, i32 8, i32 4, i32 2>
  %shuffle1 = shufflevector <16 x i64> %shuffle, <16 x i64> zeroinitializer, <16 x i32> <i32 1, i32 5, i32 4, i32 2, i32 7, i32 13, i32 3, i32 2, i32 5, i32 10, i32 8, i32 4, i32 0, i32 8, i32 7, i32 6>
  store <16 x i64> %shuffle1, ptr null, align 128
  ret i32 0
}

attributes #0 = { "target-features"="+avx2" }
