; Ensure assertion is not hit when folding concat of two contiguous extract_subvector operations
; from a source with a non-power-of-two vector length.
; RUN: llc -mattr=+avx2 < %s

source_filename = "foo.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr noundef %pDst, ptr noundef %pSrc) {
bb0:
  %sptr1 = getelementptr i8, ptr %pSrc, i64 32
  %load598 = load <12 x float>, ptr %sptr1, align 1
  br label %bb1
bb1:
  %sptr0 = getelementptr i8, ptr %pSrc, i64 16
  %load617 = load <12 x float>, ptr %sptr0, align 1
  %42 = fsub contract <12 x float> %load617, %load598
  %43 = shufflevector <12 x float> %42, <12 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %44 = fsub contract <12 x float> %load617, %load598
  %45 = shufflevector <12 x float> %44, <12 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %46 = fsub contract <12 x float> %load617, %load598
  %47 = shufflevector <12 x float> %46, <12 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %dptr0 = getelementptr i8, ptr %pDst, i64 16
  %dptr1 = getelementptr i8, ptr %pDst, i64 32 
  %dptr2 = getelementptr i8, ptr %pDst, i64 48
  store <4 x float> %43, ptr %dptr0, align 1
  store <4 x float> %45, ptr %dptr1, align 1
  store <4 x float> %47, ptr %dptr2, align 1
  ret void
}
