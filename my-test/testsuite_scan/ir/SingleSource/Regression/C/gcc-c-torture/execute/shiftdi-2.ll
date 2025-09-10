; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/shiftdi-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/shiftdi-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i64 568513516876543756, align 8
@b = dso_local local_unnamed_addr global i64 -754324895235774564, align 8
@c = dso_local local_unnamed_addr global i64 156789543257562457, align 8
@expected_a = dso_local local_unnamed_addr global [64 x i64] [i64 568513516876543756, i64 1137027033753087512, i64 2274054067506175024, i64 4548108135012350048, i64 9096216270024700096, i64 -254311533660151424, i64 -508623067320302848, i64 -1017246134640605696, i64 -2034492269281211392, i64 -4068984538562422784, i64 -8137969077124845568, i64 2170805919459860480, i64 4341611838919720960, i64 8683223677839441920, i64 -1080296718030667776, i64 -2160593436061335552, i64 -4321186872122671104, i64 -8642373744245342208, i64 1161996585218867200, i64 2323993170437734400, i64 4647986340875468800, i64 -9150771391958614016, i64 145201289792323584, i64 290402579584647168, i64 580805159169294336, i64 1161610318338588672, i64 2323220636677177344, i64 4646441273354354688, i64 -9153861527000842240, i64 139021019707867136, i64 278042039415734272, i64 556084078831468544, i64 1112168157662937088, i64 2224336315325874176, i64 4448672630651748352, i64 8897345261303496704, i64 -652053551102558208, i64 -1304107102205116416, i64 -2608214204410232832, i64 -5216428408820465664, i64 8013887256068620288, i64 -2418969561572311040, i64 -4837939123144622080, i64 8770865827420307456, i64 -905012418868936704, i64 -1810024837737873408, i64 -3620049675475746816, i64 -7240099350951493632, i64 3966545371806564352, i64 7933090743613128704, i64 -2580562586483294208, i64 -5161125172966588416, i64 8124493727776374784, i64 -2197756618156802048, i64 -4395513236313604096, i64 -8791026472627208192, i64 864691128455135232, i64 1729382256910270464, i64 3458764513820540928, i64 6917529027641081856, i64 -4611686018427387904, i64 -9223372036854775808, i64 0, i64 0], align 8
@expected_b = dso_local local_unnamed_addr global [64 x i64] [i64 -754324895235774564, i64 -377162447617887282, i64 -188581223808943641, i64 -94290611904471821, i64 -47145305952235911, i64 -23572652976117956, i64 -11786326488058978, i64 -5893163244029489, i64 -2946581622014745, i64 -1473290811007373, i64 -736645405503687, i64 -368322702751844, i64 -184161351375922, i64 -92080675687961, i64 -46040337843981, i64 -23020168921991, i64 -11510084460996, i64 -5755042230498, i64 -2877521115249, i64 -1438760557625, i64 -719380278813, i64 -359690139407, i64 -179845069704, i64 -89922534852, i64 -44961267426, i64 -22480633713, i64 -11240316857, i64 -5620158429, i64 -2810079215, i64 -1405039608, i64 -702519804, i64 -351259902, i64 -175629951, i64 -87814976, i64 -43907488, i64 -21953744, i64 -10976872, i64 -5488436, i64 -2744218, i64 -1372109, i64 -686055, i64 -343028, i64 -171514, i64 -85757, i64 -42879, i64 -21440, i64 -10720, i64 -5360, i64 -2680, i64 -1340, i64 -670, i64 -335, i64 -168, i64 -84, i64 -42, i64 -21, i64 -11, i64 -6, i64 -3, i64 -2, i64 -1, i64 -1, i64 -1, i64 -1], align 8
@expected_c = dso_local local_unnamed_addr global [64 x i64] [i64 156789543257562457, i64 78394771628781228, i64 39197385814390614, i64 19598692907195307, i64 9799346453597653, i64 4899673226798826, i64 2449836613399413, i64 1224918306699706, i64 612459153349853, i64 306229576674926, i64 153114788337463, i64 76557394168731, i64 38278697084365, i64 19139348542182, i64 9569674271091, i64 4784837135545, i64 2392418567772, i64 1196209283886, i64 598104641943, i64 299052320971, i64 149526160485, i64 74763080242, i64 37381540121, i64 18690770060, i64 9345385030, i64 4672692515, i64 2336346257, i64 1168173128, i64 584086564, i64 292043282, i64 146021641, i64 73010820, i64 36505410, i64 18252705, i64 9126352, i64 4563176, i64 2281588, i64 1140794, i64 570397, i64 285198, i64 142599, i64 71299, i64 35649, i64 17824, i64 8912, i64 4456, i64 2228, i64 1114, i64 557, i64 278, i64 139, i64 69, i64 34, i64 17, i64 8, i64 4, i64 2, i64 1, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i64, ptr @a, align 8, !tbaa !6
  %2 = load i64, ptr @b, align 8
  %3 = load i64, ptr @c, align 8
  br label %7

4:                                                ; preds = %18
  %5 = add nuw nsw i64 %8, 1
  %6 = icmp eq i64 %5, 64
  br i1 %6, label %24, label %7, !llvm.loop !10

7:                                                ; preds = %0, %4
  %8 = phi i64 [ 0, %0 ], [ %5, %4 ]
  %9 = shl i64 %1, %8
  %10 = getelementptr inbounds nuw i64, ptr @expected_a, i64 %8
  %11 = load i64, ptr %10, align 8, !tbaa !6
  %12 = icmp eq i64 %9, %11
  br i1 %12, label %13, label %23

13:                                               ; preds = %7
  %14 = ashr i64 %2, %8
  %15 = getelementptr inbounds nuw i64, ptr @expected_b, i64 %8
  %16 = load i64, ptr %15, align 8, !tbaa !6
  %17 = icmp eq i64 %14, %16
  br i1 %17, label %18, label %23

18:                                               ; preds = %13
  %19 = lshr i64 %3, %8
  %20 = getelementptr inbounds nuw i64, ptr @expected_c, i64 %8
  %21 = load i64, ptr %20, align 8, !tbaa !6
  %22 = icmp eq i64 %19, %21
  br i1 %22, label %4, label %23

23:                                               ; preds = %18, %13, %7
  tail call void @abort() #2
  unreachable

24:                                               ; preds = %4
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
