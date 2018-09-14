; RUN: llc < %s  -mtriple=x86_64-apple-osx10.14 -mattr=+avx2 | FileCheck %s

; Check that we don't crash due creating invalid extract_subvector indices in store merging.
; CHECK-LABEL: testfn
; CHECK: retq
define void @testfn(i32* nocapture %p) {
  %v0 = getelementptr i32, i32* %p, i64 12
  %1 = bitcast i32* %v0 to <2 x i64>*
  %2 = bitcast i32* %v0 to <4 x i32>*
  %3 = getelementptr <2 x i64>, <2 x i64>* %1, i64 -3
  store <2 x i64> undef, <2 x i64>* %3, align 16
  %4 = shufflevector <4 x i64> zeroinitializer, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
  %5 = getelementptr <2 x i64>, <2 x i64>* %1, i64 -2
  store <2 x i64> %4, <2 x i64>* %5, align 16
  %6 = shufflevector <8 x i32> zeroinitializer, <8 x i32> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %7 = getelementptr <4 x i32>, <4 x i32>* %2, i64 -1
  store <4 x i32> %6, <4 x i32>* %7, align 16
  ret void
}
