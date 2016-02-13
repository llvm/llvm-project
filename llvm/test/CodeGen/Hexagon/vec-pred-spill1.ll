; RUN: llc -march=hexagon -mcpu=hexagonv60 -O2 -enable-hexagon-hvx < %s | FileCheck %s

; CHECK: vmem(r{{[0-9]+}}+#3) = v{{[0-9]+}}
; CHECK: call puts
; CHECK: call print_vecpred
; CHECK: v{{[0-9]+}}{{ *}}={{ *}}vmem(r{{[0-9]+}}+#3)

target triple = "hexagon"

@K = global i64 0, align 8
@src = global i32 -1, align 4
@Q6VecPredResult = common global <16 x i32> zeroinitializer, align 64
@dst_addresses = common global [15 x i64] zeroinitializer, align 8
@ptr_addresses = common global [15 x i8*] zeroinitializer, align 8
@src_addresses = common global [15 x i8*] zeroinitializer, align 8
@ptr = common global [32768 x i32] zeroinitializer, align 8
@vecpreds = common global [15 x <16 x i32>] zeroinitializer, align 64
@VectorResult = common global <16 x i32> zeroinitializer, align 64
@vectors = common global [15 x <16 x i32>] zeroinitializer, align 64
@VectorPairResult = common global <32 x i32> zeroinitializer, align 128
@vector_pairs = common global [15 x <32 x i32>] zeroinitializer, align 128
@str = private unnamed_addr constant [106 x i8] c"Q6VecPred4 :  Q6_Q_vandor_QVR(Q6_Q_vand_VR(Q6_V_vsplat_R(1+1),(0x01010101)),Q6_V_vsplat_R(0+1),INT32_MIN)\00"
@str3 = private unnamed_addr constant [99 x i8] c"Q6VecPred4 :  Q6_Q_vandor_QVR(Q6_Q_vand_VR(Q6_V_vsplat_R(1+1),(0x01010101)),Q6_V_vsplat_R(0+1),-1)\00"
@str4 = private unnamed_addr constant [98 x i8] c"Q6VecPred4 :  Q6_Q_vandor_QVR(Q6_Q_vand_VR(Q6_V_vsplat_R(1+1),(0x01010101)),Q6_V_vsplat_R(0+1),0)\00"

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %call = tail call i32 bitcast (i32 (...)* @init_addresses to i32 ()*)() #3
  %call1 = tail call i32 @acquire_vector_unit(i8 zeroext 0) #3
  tail call void @init_vectors() #3
  %0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2)
  %1 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %0, i32 16843009)
  %2 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %3 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt.acc(<512 x i1> %1, <16 x i32> %2, i32 -2147483648)
  %4 = bitcast <512 x i1> %3 to <16 x i32>
  store <16 x i32> %4, <16 x i32>* @Q6VecPredResult, align 64, !tbaa !1
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([106 x i8], [106 x i8]* @str, i32 0, i32 0))
  tail call void @print_vecpred(i32 512, i8* bitcast (<16 x i32>* @Q6VecPredResult to i8*)) #3
  %5 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt.acc(<512 x i1> %1, <16 x i32> %2, i32 -1)
  %6 = bitcast <512 x i1> %5 to <16 x i32>
  store <16 x i32> %6, <16 x i32>* @Q6VecPredResult, align 64, !tbaa !1
  %puts5 = tail call i32 @puts(i8* getelementptr inbounds ([99 x i8], [99 x i8]* @str3, i32 0, i32 0))
  tail call void @print_vecpred(i32 512, i8* bitcast (<16 x i32>* @Q6VecPredResult to i8*)) #3
  %7 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt.acc(<512 x i1> %1, <16 x i32> %2, i32 0)
  %8 = bitcast <512 x i1> %7 to <16 x i32>
  store <16 x i32> %8, <16 x i32>* @Q6VecPredResult, align 64, !tbaa !1
  %puts6 = tail call i32 @puts(i8* getelementptr inbounds ([98 x i8], [98 x i8]* @str4, i32 0, i32 0))
  tail call void @print_vecpred(i32 512, i8* bitcast (<16 x i32>* @Q6VecPredResult to i8*)) #3
  ret i32 0
}

declare i32 @init_addresses(...) #1

declare i32 @acquire_vector_unit(i8 zeroext) #1

declare void @init_vectors() #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vandvrt.acc(<512 x i1>, <16 x i32>, i32) #2

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #2

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #2

declare void @print_vecpred(i32, i8*) #1

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
