; ModuleID = 'llvm/test/Transforms/InstCombine/pblend.ll'
source_filename = "llvm/test/Transforms/InstCombine/pblend.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define <2 x i64> @tricky(<2 x i64> noundef %a, <2 x i64> noundef %b, <2 x i64> noundef %c, <2 x i64> noundef %src) {
entry:
  %0 = bitcast <2 x i64> %a to <4 x i32>
  %cmp.i = icmp sgt <4 x i32> %0, zeroinitializer
  %sext.i = sext <4 x i1> %cmp.i to <4 x i32>
  %1 = bitcast <4 x i32> %sext.i to <2 x i64>
  %2 = bitcast <2 x i64> %b to <4 x i32>
  %cmp.i21 = icmp sgt <4 x i32> %2, zeroinitializer
  %sext.i22 = sext <4 x i1> %cmp.i21 to <4 x i32>
  %3 = bitcast <4 x i32> %sext.i22 to <2 x i64>
  %4 = bitcast <2 x i64> %c to <4 x i32>
  %cmp.i23 = icmp sgt <4 x i32> %4, zeroinitializer
  %sext.i24 = sext <4 x i1> %cmp.i23 to <4 x i32>
  %5 = bitcast <4 x i32> %sext.i24 to <2 x i64>
  %and.i = and <2 x i64> %3, %1
  %xor.i = xor <2 x i64> %and.i, %5
  %and.i25 = and <2 x i64> %xor.i, %1
  %and.i26 = and <2 x i64> %xor.i, %3
  %and.i27 = and <2 x i64> %and.i, %src
  %6 = bitcast <2 x i64> %and.i27 to <16 x i8>
  %7 = bitcast <2 x i64> %a to <16 x i8>
  %8 = bitcast <2 x i64> %and.i25 to <16 x i8>
  %9 = tail call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %6, <16 x i8> %7, <16 x i8> %8)
  %10 = bitcast <2 x i64> %b to <16 x i8>
  %11 = bitcast <2 x i64> %and.i26 to <16 x i8>
  %12 = tail call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %9, <16 x i8> %10, <16 x i8> %11)
  %13 = bitcast <16 x i8> %12 to <2 x i64>
  ret <2 x i64> %13
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
