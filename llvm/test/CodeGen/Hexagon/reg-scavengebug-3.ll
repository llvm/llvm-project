; RUN: llc -O0 -mtriple=hexagon -mcpu=hexagonv60 -mattr=+hvxv60,hvx-length64b < %s | FileCheck %s

; CHECK: vmem

target triple = "hexagon"

@vecpreds = external global [15 x <16 x i32>], align 64
@vectors = external global [15 x <16 x i32>], align 64
@vector_pairs = external global [15 x <32 x i32>], align 128
@.str1 = external hidden unnamed_addr constant [20 x i8], align 1
@.str2 = external hidden unnamed_addr constant [43 x i8], align 1
@Q6VecPredResult = external global <16 x i32>, align 64
@.str52 = external hidden unnamed_addr constant [57 x i8], align 1
@.str54 = external hidden unnamed_addr constant [59 x i8], align 1
@VectorResult = external global <16 x i32>, align 64
@.str243 = external hidden unnamed_addr constant [60 x i8], align 1
@.str251 = external hidden unnamed_addr constant [77 x i8], align 1
@.str290 = external hidden unnamed_addr constant [65 x i8], align 1
@VectorPairResult = external global <32 x i32>, align 128

; Function Attrs: nounwind
declare void @print_vector(i32, ptr) #0

; Function Attrs: nounwind
declare i32 @printf(ptr, ...) #0

; Function Attrs: nounwind
declare void @print_vecpred(i32, ptr) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1

; Function Attrs: nounwind
declare void @init_vectors() #0

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind
declare void @init_addresses() #0

; Function Attrs: nounwind
declare <16 x i32> @llvm.hexagon.V6.vsubhnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %0 = load <16 x i32>, ptr @vecpreds, align 64
  %1 = load <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vectors, i32 0, i32 1), align 64
  call void @print_vecpred(i32 64, ptr @Q6VecPredResult)
  %2 = load <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vectors, i32 0, i32 1), align 64
  %call50 = call i32 (ptr, ...) @printf(ptr @.str52) #3
  %3 = load <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vectors, i32 0, i32 1), align 64
  %call52 = call i32 (ptr, ...) @printf(ptr @.str54) #3
  %4 = load <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vectors, i32 0, i32 1), align 64
  %call300 = call i32 (ptr, ...) @printf(ptr @.str290) #3
  %5 = load <16 x i32>, ptr @vectors, align 64
  %6 = load <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @vectors, i32 0, i32 1), align 64
  %call1373 = call i32 (ptr, ...) @printf(ptr @.str1, ptr @.str2, ptr @.str243) #3
  %7 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %call1381 = call i32 (ptr, ...) @printf(ptr @.str1, ptr @.str2, ptr @.str251) #3
  %8 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %9 = call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %8, i32 16843009)
  call void @print_vector(i32 64, ptr @VectorResult)
  %10 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %11 = call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %10, i32 16843009)
  %12 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %11, i32 -1)
  %13 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %12, i32 -1)
  %14 = call <16 x i32> @llvm.hexagon.V6.vsubhnq(<64 x i1> %13, <16 x i32> undef, <16 x i32> undef)
  store <16 x i32> %14, ptr @VectorResult, align 64
  ret i32 0
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
