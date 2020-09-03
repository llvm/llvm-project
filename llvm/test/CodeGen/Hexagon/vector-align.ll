; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that the store to Q6VecPredResult does not get expanded into multiple
; stores. There should be no memd's. This relies on the alignment specified
; in the data layout string, so don't provide one here to make sure that the
; default one from HexagonTargetMachine is correct.

; CHECK-NOT: memd


@Q6VecPredResult = common global <16 x i32> zeroinitializer, align 64

define i32 @foo() #0 {
entry:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -2147483648)
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v1, i32 -1)
  store <16 x i32> %v2, <16 x i32>* @Q6VecPredResult, align 64, !tbaa !1
  tail call void @print_vecpred(i32 64, i8* bitcast (<16 x i32>* @Q6VecPredResult to i8*)) #3
  ret i32 0
}

declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

declare void @print_vecpred(i32, i8*) #2

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+hvxv66,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
