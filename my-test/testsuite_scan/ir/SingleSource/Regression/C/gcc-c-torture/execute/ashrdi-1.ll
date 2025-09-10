; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ashrdi-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ashrdi-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@switch.table.main = private unnamed_addr constant [64 x i64] [i64 8526495107234113920, i64 4263247553617056960, i64 2131623776808528480, i64 1065811888404264240, i64 532905944202132120, i64 266452972101066060, i64 133226486050533030, i64 66613243025266515, i64 33306621512633257, i64 16653310756316628, i64 8326655378158314, i64 4163327689079157, i64 2081663844539578, i64 1040831922269789, i64 520415961134894, i64 260207980567447, i64 130103990283723, i64 65051995141861, i64 32525997570930, i64 16262998785465, i64 8131499392732, i64 4065749696366, i64 2032874848183, i64 1016437424091, i64 508218712045, i64 254109356022, i64 127054678011, i64 63527339005, i64 31763669502, i64 15881834751, i64 7940917375, i64 3970458687, i64 1985229343, i64 992614671, i64 496307335, i64 248153667, i64 124076833, i64 62038416, i64 31019208, i64 15509604, i64 7754802, i64 3877401, i64 1938700, i64 969350, i64 484675, i64 242337, i64 121168, i64 60584, i64 30292, i64 15146, i64 7573, i64 3786, i64 1893, i64 946, i64 473, i64 236, i64 118, i64 59, i64 29, i64 14, i64 7, i64 3, i64 1, i64 0], align 8
@switch.table.main.1 = private unnamed_addr constant [64 x i64] [i64 -8152436031399644656, i64 -4076218015699822328, i64 -2038109007849911164, i64 -1019054503924955582, i64 -509527251962477791, i64 -254763625981238896, i64 -127381812990619448, i64 -63690906495309724, i64 -31845453247654862, i64 -15922726623827431, i64 -7961363311913716, i64 -3980681655956858, i64 -1990340827978429, i64 -995170413989215, i64 -497585206994608, i64 -248792603497304, i64 -124396301748652, i64 -62198150874326, i64 -31099075437163, i64 -15549537718582, i64 -7774768859291, i64 -3887384429646, i64 -1943692214823, i64 -971846107412, i64 -485923053706, i64 -242961526853, i64 -121480763427, i64 -60740381714, i64 -30370190857, i64 -15185095429, i64 -7592547715, i64 -3796273858, i64 -1898136929, i64 -949068465, i64 -474534233, i64 -237267117, i64 -118633559, i64 -59316780, i64 -29658390, i64 -14829195, i64 -7414598, i64 -3707299, i64 -1853650, i64 -926825, i64 -463413, i64 -231707, i64 -115854, i64 -57927, i64 -28964, i64 -14482, i64 -7241, i64 -3621, i64 -1811, i64 -906, i64 -453, i64 -227, i64 -114, i64 -57, i64 -29, i64 -15, i64 -8, i64 -4, i64 -2, i64 -1], align 8

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %10, %1 ]
  %3 = phi <2 x i64> [ <i64 0, i64 1>, %0 ], [ %15, %1 ]
  %4 = lshr <2 x i64> splat (i64 8526495107234113920), %3
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
  br i1 %12, label %33, label %17

17:                                               ; preds = %16, %17
  %18 = phi i64 [ %26, %17 ], [ 0, %16 ]
  %19 = phi <2 x i64> [ %31, %17 ], [ <i64 0, i64 1>, %16 ]
  %20 = ashr <2 x i64> splat (i64 -8152436031399644656), %19
  %21 = getelementptr inbounds nuw i64, ptr @switch.table.main.1, i64 %18
  %22 = load <2 x i64>, ptr %21, align 8, !tbaa !6
  %23 = freeze <2 x i64> %20
  %24 = freeze <2 x i64> %22
  %25 = icmp ne <2 x i64> %23, %24
  %26 = add nuw i64 %18, 2
  %27 = bitcast <2 x i1> %25 to i2
  %28 = icmp ne i2 %27, 0
  %29 = icmp eq i64 %26, 64
  %30 = or i1 %28, %29
  %31 = add <2 x i64> %19, splat (i64 2)
  br i1 %30, label %32, label %17, !llvm.loop !14

32:                                               ; preds = %17
  br i1 %28, label %34, label %38

33:                                               ; preds = %16
  tail call void @abort() #3
  unreachable

34:                                               ; preds = %32
  tail call void @abort() #3
  unreachable

35:                                               ; preds = %38
  %36 = add nuw nsw i64 %39, 1
  %37 = icmp eq i64 %36, 64
  br i1 %37, label %50, label %38, !llvm.loop !15

38:                                               ; preds = %32, %35
  %39 = phi i64 [ %36, %35 ], [ 0, %32 ]
  %40 = and i64 %39, 4294967295
  %41 = getelementptr inbounds nuw i64, ptr @switch.table.main, i64 %40
  %42 = load i64, ptr %41, align 8
  %43 = getelementptr inbounds nuw i64, ptr @switch.table.main, i64 %39
  %44 = load i64, ptr %43, align 8, !tbaa !6
  %45 = icmp eq i64 %42, %44
  br i1 %45, label %35, label %46

46:                                               ; preds = %38
  tail call void @abort() #3
  unreachable

47:                                               ; preds = %50
  %48 = add nuw nsw i64 %51, 1
  %49 = icmp eq i64 %48, 64
  br i1 %49, label %59, label %50, !llvm.loop !16

50:                                               ; preds = %35, %47
  %51 = phi i64 [ %48, %47 ], [ 0, %35 ]
  %52 = and i64 %51, 4294967295
  %53 = getelementptr inbounds nuw i64, ptr @switch.table.main.1, i64 %52
  %54 = load i64, ptr %53, align 8
  %55 = getelementptr inbounds nuw i64, ptr @switch.table.main.1, i64 %51
  %56 = load i64, ptr %55, align 8, !tbaa !6
  %57 = icmp eq i64 %54, %56
  br i1 %57, label %47, label %58

58:                                               ; preds = %50
  tail call void @abort() #3
  unreachable

59:                                               ; preds = %47
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
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11}
!16 = distinct !{!16, !11}
