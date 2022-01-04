; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Test that the HexagonVectorCombine pass does not move an instruction
; incorrectly, which causes a GEP to have a base that is not defined.
; If the pass runs correctly, the unaligned loads are converted to
; aligned loads instead of crashing.

; CHECK-NOT: vmemu

define dllexport void @test() local_unnamed_addr #0 {
entry:
  br label %for_begin77

for_begin77:
  %0 = load i8*, i8** undef, align 4
  %1 = getelementptr i8, i8* %0, i32 1794
  %2 = bitcast i8* %1 to <64 x half>*
  %3 = call <64 x half> @llvm.masked.load.v64f16.p0v64f16(<64 x half>* %2, i32 1, <64 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true>, <64 x half> undef)
  %4 = getelementptr i8, i8* %0, i32 1922
  %5 = bitcast i8* %4 to <64 x half>*
  %6 = call <64 x half> @llvm.masked.load.v64f16.p0v64f16(<64 x half>* %5, i32 1, <64 x i1> <i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 false>, <64 x half> undef)
  %7 = shufflevector <64 x half> %3, <64 x half> %6, <64 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62, i32 63, i32 65, i32 67, i32 69, i32 71, i32 73, i32 75, i32 77, i32 79, i32 81, i32 83, i32 85, i32 87, i32 89, i32 91, i32 93, i32 95, i32 97, i32 99, i32 101, i32 103, i32 105, i32 107, i32 109, i32 111, i32 113, i32 115, i32 117, i32 119, i32 121, i32 123, i32 125>
  call void @llvm.assume(i1 true) [ "align"(i8* null, i32 128) ]
  %8 = getelementptr i8, i8* null, i32 128
  %9 = bitcast i8* %8 to <64 x half>*
  %10 = fadd <64 x half> zeroinitializer, %7
  %11 = shufflevector <64 x half> %10, <64 x half> undef, <64 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  %12 = getelementptr i8, i8* %0, i32 1920
  %13 = bitcast i8* %12 to <64 x half>*
  %unmaskedload243 = load <64 x half>, <64 x half>* %13, align 128
  %14 = fadd <64 x half> %11, %unmaskedload243
  store <64 x half> %14, <64 x half>* %9, align 128
  br label %for_begin77
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: argmemonly nofree nosync nounwind readonly willreturn
declare <64 x half> @llvm.masked.load.v64f16.p0v64f16(<64 x half>*, i32 immarg, <64 x i1>, <64 x half>) #2

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hvx-qfloat" }
attributes #1 = { nofree nosync nounwind willreturn }
attributes #2 = { argmemonly nofree nosync nounwind readonly willreturn }
