; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [8 x [2 x i32]], [8 x [2 x i32]], [4 x [4 x i32]], i32, i32, i32, i32, [256 x i8], [256 x i8], [256 x i8], [256 x i8], [256 x i8], i32, i32, i32, i32, i32, i32, [500 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [256 x i8], [256 x i8], [256 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1024 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [256 x i8], [256 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [256 x i8], i32, i32, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, double, double, double, [5 x double], i32, [8 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [6 x double], [6 x double], [256 x i8], i32, i32, i32, i32, [2 x [5 x i32]], [2 x [5 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], i32 }

; The execution context of invalid loads in this test case has at some point become very complex and we should bail.
; CHECK-NOT: Statements

@global = external global [300 x i8], align 16
@global1 = external global ptr, align 8
@global2 = external unnamed_addr constant [79 x i8], align 1
@global3 = external unnamed_addr constant [57 x i8], align 1

declare void @widget() #0

; Function Attrs: nounwind
declare void @quux(ptr, i64, ptr, ...) #1

; Function Attrs: nounwind uwtable
define void @hoge(ptr %A) #2 {
bb:
  br label %bb15

bb15:                                             ; preds = %bb
  %tmp = load ptr, ptr @global1, align 8, !tbaa !1
  %tmp16 = getelementptr inbounds %struct.hoge, ptr %tmp, i64 0, i32 153
  store float 1.0, ptr %A
  %tmp17 = load i32, ptr %tmp16, align 4, !tbaa !5
  %tmp18 = icmp eq i32 %tmp17, 0
  br i1 %tmp18, label %bb24, label %bb19

bb19:                                             ; preds = %bb15
  %tmp20 = getelementptr inbounds %struct.hoge, ptr %tmp, i64 0, i32 50
  store float 1.0, ptr %A
  %tmp21 = load i32, ptr %tmp20, align 8, !tbaa !9
  %tmp22 = icmp eq i32 %tmp21, 0
  br i1 %tmp22, label %bb24, label %bb23

bb23:                                             ; preds = %bb19
  call void @widget() #3
  br label %bb24

bb24:                                             ; preds = %bb23, %bb19, %bb15
  %tmp25 = load ptr, ptr @global1, align 8, !tbaa !1
  store float 1.0, ptr %A
  %tmp26 = getelementptr inbounds %struct.hoge, ptr %tmp25, i64 0, i32 16
  %tmp27 = load i32, ptr %tmp26, align 8, !tbaa !10
  %tmp28 = icmp eq i32 %tmp27, 3
  br i1 %tmp28, label %bb29, label %bb34

bb29:                                             ; preds = %bb24
  store float 1.0, ptr %A
  %tmp31 = load i32, ptr %tmp25, align 8, !tbaa !11
  %tmp32 = icmp slt i32 %tmp31, 144
  br i1 %tmp32, label %bb33, label %bb34

bb33:                                             ; preds = %bb29
  call void (ptr, i64, ptr, ...) @quux(ptr @global, i64 300, ptr @global2, i32 144) #3
  br label %bb34

bb34:                                             ; preds = %bb33, %bb29, %bb24
  %tmp35 = load ptr, ptr @global1, align 8, !tbaa !1
  store float 1.0, ptr %A
  %tmp36 = getelementptr inbounds %struct.hoge, ptr %tmp35, i64 0, i32 40
  %tmp37 = load i32, ptr %tmp36, align 8, !tbaa !12
  %tmp38 = icmp eq i32 %tmp37, 0
  br i1 %tmp38, label %bb49, label %bb39

bb39:                                             ; preds = %bb34
  %tmp40 = getelementptr inbounds %struct.hoge, ptr %tmp35, i64 0, i32 46
  store float 1.0, ptr %A
  %tmp41 = load i32, ptr %tmp40, align 8, !tbaa !13
  %tmp42 = icmp eq i32 %tmp41, 0
  br i1 %tmp42, label %bb49, label %bb43

bb43:                                             ; preds = %bb39
  %tmp44 = getelementptr inbounds %struct.hoge, ptr %tmp35, i64 0, i32 7
  store float 1.0, ptr %A
  %tmp45 = getelementptr inbounds %struct.hoge, ptr %tmp35, i64 0, i32 48
  %tmp46 = load i32, ptr %tmp45, align 8, !tbaa !14
  %tmp47 = icmp slt i32 0, %tmp46
  br i1 %tmp47, label %bb48, label %bb49

bb48:                                             ; preds = %bb43
  call void @widget() #3
  br label %bb49

bb49:                                             ; preds = %bb48, %bb43, %bb39, %bb34
  store float 1.0, ptr %A
  %tmp50 = load ptr, ptr @global1, align 8, !tbaa !1
  %tmp51 = getelementptr inbounds %struct.hoge, ptr %tmp50, i64 0, i32 198
  %tmp52 = load i32, ptr %tmp51, align 8, !tbaa !15
  %tmp53 = icmp eq i32 %tmp52, 0
  br i1 %tmp53, label %bb59, label %bb54

bb54:                                             ; preds = %bb49
  store float 1.0, ptr %A
  %tmp55 = getelementptr inbounds %struct.hoge, ptr %tmp50, i64 0, i32 16
  %tmp56 = load i32, ptr %tmp55, align 8, !tbaa !10
  %tmp57 = icmp eq i32 %tmp56, 0
  br i1 %tmp57, label %bb58, label %bb59

bb58:                                             ; preds = %bb54
  call void (ptr, i64, ptr, ...) @quux(ptr @global, i64 300, ptr @global3) #3
  br label %bb59

bb59:                                             ; preds = %bb58, %bb54, %bb49
  store float 1.0, ptr %A
  %tmp60 = load ptr, ptr @global1, align 8, !tbaa !1
  %tmp61 = getelementptr inbounds %struct.hoge, ptr %tmp60, i64 0, i32 31
  %tmp62 = load i32, ptr %tmp61, align 4, !tbaa !16
  %tmp63 = icmp eq i32 %tmp62, 0
  br i1 %tmp63, label %bb65, label %bb64

bb64:                                             ; preds = %bb59
  br label %bb65

bb65:                                             ; preds = %bb64, %bb59
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 252261) (llvm/trunk 252271)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !7, i64 5100}
!6 = !{!"", !7, i64 0, !7, i64 4, !7, i64 8, !7, i64 12, !7, i64 16, !7, i64 20, !7, i64 24, !7, i64 28, !7, i64 32, !7, i64 36, !7, i64 40, !7, i64 44, !7, i64 48, !7, i64 52, !7, i64 56, !7, i64 60, !7, i64 64, !7, i64 68, !3, i64 72, !3, i64 136, !3, i64 200, !7, i64 264, !7, i64 268, !7, i64 272, !7, i64 276, !3, i64 280, !3, i64 536, !3, i64 792, !3, i64 1048, !3, i64 1304, !7, i64 1560, !7, i64 1564, !7, i64 1568, !7, i64 1572, !7, i64 1576, !7, i64 1580, !3, i64 1584, !7, i64 2084, !7, i64 2088, !7, i64 2092, !7, i64 2096, !7, i64 2100, !7, i64 2104, !7, i64 2108, !7, i64 2112, !7, i64 2116, !7, i64 2120, !7, i64 2124, !7, i64 2128, !7, i64 2132, !7, i64 2136, !7, i64 2140, !7, i64 2144, !7, i64 2148, !7, i64 2152, !7, i64 2156, !3, i64 2160, !3, i64 2416, !3, i64 2672, !7, i64 2928, !7, i64 2932, !7, i64 2936, !7, i64 2940, !7, i64 2944, !7, i64 2948, !7, i64 2952, !7, i64 2956, !7, i64 2960, !7, i64 2964, !7, i64 2968, !7, i64 2972, !3, i64 2976, !7, i64 4000, !7, i64 4004, !7, i64 4008, !7, i64 4012, !7, i64 4016, !7, i64 4020, !7, i64 4024, !7, i64 4028, !7, i64 4032, !7, i64 4036, !7, i64 4040, !7, i64 4044, !7, i64 4048, !7, i64 4052, !7, i64 4056, !7, i64 4060, !7, i64 4064, !7, i64 4068, !7, i64 4072, !7, i64 4076, !8, i64 4080, !7, i64 4088, !7, i64 4092, !7, i64 4096, !7, i64 4100, !7, i64 4104, !7, i64 4108, !7, i64 4112, !7, i64 4116, !7, i64 4120, !7, i64 4124, !7, i64 4128, !7, i64 4132, !7, i64 4136, !7, i64 4140, !7, i64 4144, !7, i64 4148, !7, i64 4152, !7, i64 4156, !7, i64 4160, !7, i64 4164, !7, i64 4168, !7, i64 4172, !7, i64 4176, !7, i64 4180, !7, i64 4184, !7, i64 4188, !3, i64 4192, !3, i64 4448, !7, i64 4704, !7, i64 4708, !7, i64 4712, !7, i64 4716, !7, i64 4720, !7, i64 4724, !7, i64 4728, !7, i64 4732, !7, i64 4736, !7, i64 4740, !7, i64 4744, !7, i64 4748, !7, i64 4752, !7, i64 4756, !7, i64 4760, !7, i64 4764, !7, i64 4768, !7, i64 4772, !3, i64 4776, !7, i64 5032, !7, i64 5036, !2, i64 5040, !2, i64 5048, !2, i64 5056, !2, i64 5064, !7, i64 5072, !7, i64 5076, !7, i64 5080, !7, i64 5084, !7, i64 5088, !7, i64 5092, !7, i64 5096, !7, i64 5100, !7, i64 5104, !7, i64 5108, !7, i64 5112, !7, i64 5116, !7, i64 5120, !7, i64 5124, !7, i64 5128, !7, i64 5132, !7, i64 5136, !8, i64 5144, !8, i64 5152, !8, i64 5160, !3, i64 5168, !7, i64 5208, !3, i64 5212, !3, i64 5244, !7, i64 5248, !7, i64 5252, !7, i64 5256, !7, i64 5260, !7, i64 5264, !7, i64 5268, !7, i64 5272, !7, i64 5276, !7, i64 5280, !7, i64 5284, !7, i64 5288, !3, i64 5296, !3, i64 5344, !3, i64 5392, !7, i64 5648, !7, i64 5652, !7, i64 5656, !7, i64 5660, !3, i64 5664, !3, i64 5704, !7, i64 5744, !7, i64 5748, !7, i64 5752, !7, i64 5756, !7, i64 5760, !7, i64 5764, !7, i64 5768, !7, i64 5772, !7, i64 5776, !3, i64 5780, !7, i64 5792}
!7 = !{!"int", !3, i64 0}
!8 = !{!"double", !3, i64 0}
!9 = !{!6, !7, i64 2136}
!10 = !{!6, !7, i64 64}
!11 = !{!6, !7, i64 0}
!12 = !{!6, !7, i64 2096}
!13 = !{!6, !7, i64 2120}
!14 = !{!6, !7, i64 2128}
!15 = !{!6, !7, i64 5776}
!16 = !{!6, !7, i64 1564}
