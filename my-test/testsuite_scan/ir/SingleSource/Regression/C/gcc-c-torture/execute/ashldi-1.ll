; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ashldi-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ashldi-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@switch.table.main = private unnamed_addr constant [64 x i64] [i64 81985529216486895, i64 163971058432973790, i64 327942116865947580, i64 655884233731895160, i64 1311768467463790320, i64 2623536934927580640, i64 5247073869855161280, i64 -7952596333999229056, i64 2541551405711093504, i64 5083102811422187008, i64 -8280538450865177600, i64 1885667171979196416, i64 3771334343958392832, i64 7542668687916785664, i64 -3361406697875980288, i64 -6722813395751960576, i64 5001117282205630464, i64 -8444509509298290688, i64 1557725055112970240, i64 3115450110225940480, i64 6230900220451880960, i64 -5984943632805789696, i64 6476856808097972224, i64 -5493030457513607168, i64 7460683158682337280, i64 -3525377756344877056, i64 -7050755512689754112, i64 4345233048330043392, i64 8690466096660086784, i64 -1065811880389378048, i64 -2131623760778756096, i64 -4263247521557512192, i64 -8526495043115024384, i64 1393753987479502848, i64 2787507974959005696, i64 5575015949918011392, i64 -7296712173873528832, i64 3853319725962493952, i64 7706639451924987904, i64 -3033465169859575808, i64 -6066930339719151616, i64 6312883394271248384, i64 -5820977285167054848, i64 6804789503375441920, i64 -4837165066958667776, i64 8772413939792216064, i64 -901916194125119488, i64 -1803832388250238976, i64 -3607664776500477952, i64 -7215329553000955904, i64 4016084967707639808, i64 8032169935415279616, i64 -2382404202878992384, i64 -4764808405757984768, i64 8917127262193582080, i64 -612489549322387456, i64 -1224979098644774912, i64 -2449958197289549824, i64 -4899916394579099648, i64 8646911284551352320, i64 -1152921504606846976, i64 -2305843009213693952, i64 -4611686018427387904, i64 -9223372036854775808], align 8

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %10, %1 ]
  %3 = phi <2 x i64> [ <i64 0, i64 1>, %0 ], [ %15, %1 ]
  %4 = shl <2 x i64> splat (i64 81985529216486895), %3
  %5 = getelementptr inbounds nuw i64, ptr @switch.table.main, i64 %2
  %6 = load <2 x i64>, ptr %5, align 8, !tbaa !6
  %7 = freeze <2 x i64> %4
  %8 = freeze <2 x i64> %6
  %9 = icmp ne <2 x i64> %7, %8
  %10 = add nuw i64 %2, 2
  %11 = bitcast <2 x i1> %9 to i2
  %12 = icmp ne i2 %11, 0
  %13 = icmp eq i64 %10, 64
  %14 = or i1 %12, %13
  %15 = add <2 x i64> %3, splat (i64 2)
  br i1 %14, label %16, label %1, !llvm.loop !10

16:                                               ; preds = %1
  br i1 %12, label %17, label %21

17:                                               ; preds = %16
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %21
  %19 = add nuw nsw i64 %22, 1
  %20 = icmp eq i64 %19, 64
  br i1 %20, label %30, label %21, !llvm.loop !14

21:                                               ; preds = %16, %18
  %22 = phi i64 [ %19, %18 ], [ 0, %16 ]
  %23 = and i64 %22, 4294967295
  %24 = getelementptr inbounds nuw i64, ptr @switch.table.main, i64 %23
  %25 = load i64, ptr %24, align 8
  %26 = getelementptr inbounds nuw i64, ptr @switch.table.main, i64 %22
  %27 = load i64, ptr %26, align 8, !tbaa !6
  %28 = icmp eq i64 %25, %27
  br i1 %28, label %18, label %29

29:                                               ; preds = %21
  tail call void @abort() #3
  unreachable

30:                                               ; preds = %18
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11}
