; RUN: llc -mtriple=hexagon -O2 -mcpu=hexagonv60 < %s | FileCheck %s

; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
target datalayout = "e-m:e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

@K = global i64 0, align 8
@src = global i8 -1, align 1
@vecpreds = common global [15 x <16 x i32>] zeroinitializer, align 64
@Q6VecPredResult = common global <16 x i32> zeroinitializer, align 64
@vectors = common global [15 x <16 x i32>] zeroinitializer, align 64
@VectorResult = common global <16 x i32> zeroinitializer, align 64
@vector_pairs = common global [15 x <32 x i32>] zeroinitializer, align 128
@VectorPairResult = common global <32 x i32> zeroinitializer, align 128
@dst_addresses = common global [15 x i8] zeroinitializer, align 8
@ptr_addresses = common global [15 x ptr] zeroinitializer, align 8
@src_addresses = common global [15 x ptr] zeroinitializer, align 8
@dst = common global i8 0, align 1
@ptr = common global [32768 x i8] zeroinitializer, align 8

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = load volatile <16 x i32>, ptr @vecpreds, align 64
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %0, i32 -1)
  %2 = load volatile <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vecpreds, i32 0, i32 1), align 64
  %3 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %2, i32 -1)
  %4 = call <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1> %1, <64 x i1> %3)
  %5 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %4, i32 -1)
  store volatile <16 x i32> %5, ptr @Q6VecPredResult, align 64
  %6 = load volatile <16 x i32>, ptr @vecpreds, align 64
  %7 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %6, i32 -1)
  %8 = load volatile <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vecpreds, i32 0, i32 1), align 64
  %9 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %8, i32 -1)
  %10 = call <64 x i1> @llvm.hexagon.V6.pred.and.n(<64 x i1> %7, <64 x i1> %9)
  %11 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %10, i32 -1)
  store volatile <16 x i32> %11, ptr @Q6VecPredResult, align 64
  ret i32 0

}

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1>, <64 x i1>) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.pred.and.n(<64 x i1>, <64 x i1>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
