; RUN: opt < %s -passes='print<cost-model>' -mtriple=x86_64-pc-linux-gnu 2>&1 -disable-output -mattr=+sse2 | FileCheck %s --check-prefixes=SSE2

define void @replication_i64_stride2() nounwind {
  %vf2 = shufflevector <2 x i64> undef, <2 x i64> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %vf4 = shufflevector <4 x i64> undef, <4 x i64> poison, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %vf8 = shufflevector <8 x i64> undef, <8 x i64> poison, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %vf16 = shufflevector <16 x i64> undef, <16 x i64> poison, <32 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7, i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11, i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  ret void
}
