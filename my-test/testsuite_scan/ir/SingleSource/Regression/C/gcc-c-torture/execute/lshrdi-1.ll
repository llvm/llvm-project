; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/lshrdi-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/lshrdi-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@switch.table.main = private unnamed_addr constant [64 x i64] [i64 -8690466092652643696, i64 4878138990528453960, i64 2439069495264226980, i64 1219534747632113490, i64 609767373816056745, i64 304883686908028372, i64 152441843454014186, i64 76220921727007093, i64 38110460863503546, i64 19055230431751773, i64 9527615215875886, i64 4763807607937943, i64 2381903803968971, i64 1190951901984485, i64 595475950992242, i64 297737975496121, i64 148868987748060, i64 74434493874030, i64 37217246937015, i64 18608623468507, i64 9304311734253, i64 4652155867126, i64 2326077933563, i64 1163038966781, i64 581519483390, i64 290759741695, i64 145379870847, i64 72689935423, i64 36344967711, i64 18172483855, i64 9086241927, i64 4543120963, i64 2271560481, i64 1135780240, i64 567890120, i64 283945060, i64 141972530, i64 70986265, i64 35493132, i64 17746566, i64 8873283, i64 4436641, i64 2218320, i64 1109160, i64 554580, i64 277290, i64 138645, i64 69322, i64 34661, i64 17330, i64 8665, i64 4332, i64 2166, i64 1083, i64 541, i64 270, i64 135, i64 67, i64 33, i64 16, i64 8, i64 4, i64 2, i64 1], align 8

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %10, %1 ]
  %3 = phi <2 x i64> [ <i64 0, i64 1>, %0 ], [ %15, %1 ]
  %4 = lshr <2 x i64> splat (i64 -8690466092652643696), %3
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
