; RUN: llc -march=hexagon -mcpu=hexagonv60 -enable-hexagon-hvx < %s \
; RUN:    | FileCheck %s

; Check that the store to Q6VecPredResult does not get expanded into multiple
; stores. There should be no memd's. This relies on the alignment specified
; in the data layout string, so don't provide one here to make sure that the
; default one from HexagonTargetMachine is correct.

; CHECK-NOT: memd


@Q6VecPredResult = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @foo() #0 {
entry:
  %0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %1 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %0, i32 -2147483648)
  store <512 x i1> %1, <512 x i1>* bitcast (<16 x i32>* @Q6VecPredResult to <512 x i1>*), align 64, !tbaa !1
  tail call void @print_vecpred(i32 64, i8* bitcast (<16 x i32>* @Q6VecPredResult to i8*)) #3
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

declare void @print_vecpred(i32, i8*) #2

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
