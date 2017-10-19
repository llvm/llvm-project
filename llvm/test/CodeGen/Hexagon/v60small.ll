; RUN: llc -march=hexagon -O2 -mcpu=hexagonv60  < %s | FileCheck %s

; CHECK: q{{[0-3]}} = v{{[0-9]*}}and(v{{[0-9]*}},r{{[0-9]*}})
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
@ptr_addresses = common global [15 x i8*] zeroinitializer, align 8
@src_addresses = common global [15 x i8*] zeroinitializer, align 8
@dst = common global i8 0, align 1
@ptr = common global [32768 x i8] zeroinitializer, align 8

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %1 = bitcast <16 x i32> %0 to <512 x i1>
  %2 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %3 = bitcast <16 x i32> %2 to <512 x i1>
  %4 = call <512 x i1> @llvm.hexagon.V6.pred.and(<512 x i1> %1, <512 x i1> %3)
  %5 = bitcast <512 x i1> %4 to <16 x i32>
  store volatile <16 x i32> %5, <16 x i32>* @Q6VecPredResult, align 64
  %6 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %7 = bitcast <16 x i32> %6 to <512 x i1>
  %8 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %9 = bitcast <16 x i32> %8 to <512 x i1>
  %10 = call <512 x i1> @llvm.hexagon.V6.pred.and.n(<512 x i1> %7, <512 x i1> %9)
  %11 = bitcast <512 x i1> %10 to <16 x i32>
  store volatile <16 x i32> %11, <16 x i32>* @Q6VecPredResult, align 64
  ret i32 0

}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.and(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.and.n(<512 x i1>, <512 x i1>) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
