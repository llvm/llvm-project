; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-07-VarArgs.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-07-VarArgs.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
%struct.DWordS_struct = type { i32, i8 }
%struct.LargeS_struct = type { i32, double, ptr, i32 }

@.str = private unnamed_addr constant [11 x i8] c"string %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"int %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c"double %f\0A\00", align 1
@.str.3 = private unnamed_addr constant [16 x i8] c"long long %lld\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"char %c\0A\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"DWord { %d, %c }\0A\00", align 1
@.str.6 = private unnamed_addr constant [21 x i8] c"QuadWord { %d, %f }\0A\00", align 1
@.str.7 = private unnamed_addr constant [29 x i8] c"LargeS { %d, %f, 0x%d, %d }\0A\00", align 1
@.str.8 = private unnamed_addr constant [11 x i8] c"ssiciiiiis\00", align 1
@.str.9 = private unnamed_addr constant [4 x i8] c"abc\00", align 1
@.str.10 = private unnamed_addr constant [4 x i8] c"def\00", align 1
@.str.11 = private unnamed_addr constant [14 x i8] c"10 args done!\00", align 1
@.str.12 = private unnamed_addr constant [5 x i8] c"ddil\00", align 1
@.str.13 = private unnamed_addr constant [4 x i8] c"DQL\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test(ptr noundef readonly captures(none) %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = load i8, ptr %0, align 1, !tbaa !6
  %4 = icmp eq i8 %3, 0
  br i1 %4, label %168, label %5

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 16
  br label %10

10:                                               ; preds = %5, %165
  %11 = phi i8 [ %3, %5 ], [ %166, %165 ]
  %12 = phi ptr [ %0, %5 ], [ %13, %165 ]
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 1
  switch i8 %11, label %165 [
    i8 115, label %14
    i8 105, label %31
    i8 100, label %48
    i8 108, label %65
    i8 99, label %82
    i8 68, label %100
    i8 81, label %120
    i8 76, label %139
  ]

14:                                               ; preds = %10
  %15 = load i32, ptr %6, align 8
  %16 = icmp sgt i32 %15, -1
  br i1 %16, label %24, label %17

17:                                               ; preds = %14
  %18 = add nsw i32 %15, 8
  store i32 %18, ptr %6, align 8
  %19 = icmp samesign ult i32 %15, -7
  br i1 %19, label %20, label %24

20:                                               ; preds = %17
  %21 = load ptr, ptr %7, align 8
  %22 = sext i32 %15 to i64
  %23 = getelementptr inbounds i8, ptr %21, i64 %22
  br label %27

24:                                               ; preds = %17, %14
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  store ptr %26, ptr %2, align 8
  br label %27

27:                                               ; preds = %24, %20
  %28 = phi ptr [ %23, %20 ], [ %25, %24 ]
  %29 = load ptr, ptr %28, align 8, !tbaa !9
  %30 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %29)
  br label %165

31:                                               ; preds = %10
  %32 = load i32, ptr %6, align 8
  %33 = icmp sgt i32 %32, -1
  br i1 %33, label %41, label %34

34:                                               ; preds = %31
  %35 = add nsw i32 %32, 8
  store i32 %35, ptr %6, align 8
  %36 = icmp samesign ult i32 %32, -7
  br i1 %36, label %37, label %41

37:                                               ; preds = %34
  %38 = load ptr, ptr %7, align 8
  %39 = sext i32 %32 to i64
  %40 = getelementptr inbounds i8, ptr %38, i64 %39
  br label %44

41:                                               ; preds = %34, %31
  %42 = load ptr, ptr %2, align 8
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 8
  store ptr %43, ptr %2, align 8
  br label %44

44:                                               ; preds = %41, %37
  %45 = phi ptr [ %40, %37 ], [ %42, %41 ]
  %46 = load i32, ptr %45, align 8, !tbaa !12
  %47 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %46)
  br label %165

48:                                               ; preds = %10
  %49 = load i32, ptr %8, align 4
  %50 = icmp sgt i32 %49, -1
  br i1 %50, label %58, label %51

51:                                               ; preds = %48
  %52 = add nsw i32 %49, 16
  store i32 %52, ptr %8, align 4
  %53 = icmp samesign ult i32 %49, -15
  br i1 %53, label %54, label %58

54:                                               ; preds = %51
  %55 = load ptr, ptr %9, align 8
  %56 = sext i32 %49 to i64
  %57 = getelementptr inbounds i8, ptr %55, i64 %56
  br label %61

58:                                               ; preds = %51, %48
  %59 = load ptr, ptr %2, align 8
  %60 = getelementptr inbounds nuw i8, ptr %59, i64 8
  store ptr %60, ptr %2, align 8
  br label %61

61:                                               ; preds = %58, %54
  %62 = phi ptr [ %57, %54 ], [ %59, %58 ]
  %63 = load double, ptr %62, align 8, !tbaa !14
  %64 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %63)
  br label %165

65:                                               ; preds = %10
  %66 = load i32, ptr %6, align 8
  %67 = icmp sgt i32 %66, -1
  br i1 %67, label %75, label %68

68:                                               ; preds = %65
  %69 = add nsw i32 %66, 8
  store i32 %69, ptr %6, align 8
  %70 = icmp samesign ult i32 %66, -7
  br i1 %70, label %71, label %75

71:                                               ; preds = %68
  %72 = load ptr, ptr %7, align 8
  %73 = sext i32 %66 to i64
  %74 = getelementptr inbounds i8, ptr %72, i64 %73
  br label %78

75:                                               ; preds = %68, %65
  %76 = load ptr, ptr %2, align 8
  %77 = getelementptr inbounds nuw i8, ptr %76, i64 8
  store ptr %77, ptr %2, align 8
  br label %78

78:                                               ; preds = %75, %71
  %79 = phi ptr [ %74, %71 ], [ %76, %75 ]
  %80 = load i64, ptr %79, align 8, !tbaa !16
  %81 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %80)
  br label %165

82:                                               ; preds = %10
  %83 = load i32, ptr %6, align 8
  %84 = icmp sgt i32 %83, -1
  br i1 %84, label %92, label %85

85:                                               ; preds = %82
  %86 = add nsw i32 %83, 8
  store i32 %86, ptr %6, align 8
  %87 = icmp samesign ult i32 %83, -7
  br i1 %87, label %88, label %92

88:                                               ; preds = %85
  %89 = load ptr, ptr %7, align 8
  %90 = sext i32 %83 to i64
  %91 = getelementptr inbounds i8, ptr %89, i64 %90
  br label %95

92:                                               ; preds = %85, %82
  %93 = load ptr, ptr %2, align 8
  %94 = getelementptr inbounds nuw i8, ptr %93, i64 8
  store ptr %94, ptr %2, align 8
  br label %95

95:                                               ; preds = %92, %88
  %96 = phi ptr [ %91, %88 ], [ %93, %92 ]
  %97 = load i32, ptr %96, align 8, !tbaa !12
  %98 = and i32 %97, 255
  %99 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %98)
  br label %165

100:                                              ; preds = %10
  %101 = load i32, ptr %6, align 8
  %102 = icmp sgt i32 %101, -1
  br i1 %102, label %110, label %103

103:                                              ; preds = %100
  %104 = add nsw i32 %101, 8
  store i32 %104, ptr %6, align 8
  %105 = icmp samesign ult i32 %101, -7
  br i1 %105, label %106, label %110

106:                                              ; preds = %103
  %107 = load ptr, ptr %7, align 8
  %108 = sext i32 %101 to i64
  %109 = getelementptr inbounds i8, ptr %107, i64 %108
  br label %113

110:                                              ; preds = %103, %100
  %111 = load ptr, ptr %2, align 8
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 8
  store ptr %112, ptr %2, align 8
  br label %113

113:                                              ; preds = %110, %106
  %114 = phi ptr [ %109, %106 ], [ %111, %110 ]
  %115 = load i32, ptr %114, align 8, !tbaa !12
  %116 = getelementptr inbounds nuw i8, ptr %114, i64 4
  %117 = load i8, ptr %116, align 4, !tbaa !6
  %118 = zext i8 %117 to i32
  %119 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %115, i32 noundef %118)
  br label %165

120:                                              ; preds = %10
  %121 = load i32, ptr %6, align 8
  %122 = icmp sgt i32 %121, -1
  br i1 %122, label %130, label %123

123:                                              ; preds = %120
  %124 = add nsw i32 %121, 16
  store i32 %124, ptr %6, align 8
  %125 = icmp samesign ult i32 %121, -15
  br i1 %125, label %126, label %130

126:                                              ; preds = %123
  %127 = load ptr, ptr %7, align 8
  %128 = sext i32 %121 to i64
  %129 = getelementptr inbounds i8, ptr %127, i64 %128
  br label %133

130:                                              ; preds = %123, %120
  %131 = load ptr, ptr %2, align 8
  %132 = getelementptr inbounds nuw i8, ptr %131, i64 16
  store ptr %132, ptr %2, align 8
  br label %133

133:                                              ; preds = %130, %126
  %134 = phi ptr [ %129, %126 ], [ %131, %130 ]
  %135 = load i32, ptr %134, align 8, !tbaa !12
  %136 = getelementptr inbounds nuw i8, ptr %134, i64 8
  %137 = load double, ptr %136, align 8, !tbaa !14
  %138 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef %135, double noundef %137)
  br label %165

139:                                              ; preds = %10
  %140 = load i32, ptr %6, align 8
  %141 = icmp sgt i32 %140, -1
  br i1 %141, label %149, label %142

142:                                              ; preds = %139
  %143 = add nsw i32 %140, 8
  store i32 %143, ptr %6, align 8
  %144 = icmp samesign ult i32 %140, -7
  br i1 %144, label %145, label %149

145:                                              ; preds = %142
  %146 = load ptr, ptr %7, align 8
  %147 = sext i32 %140 to i64
  %148 = getelementptr inbounds i8, ptr %146, i64 %147
  br label %152

149:                                              ; preds = %142, %139
  %150 = load ptr, ptr %2, align 8
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 8
  store ptr %151, ptr %2, align 8
  br label %152

152:                                              ; preds = %149, %145
  %153 = phi ptr [ %148, %145 ], [ %150, %149 ]
  %154 = load ptr, ptr %153, align 8
  %155 = load i32, ptr %154, align 8, !tbaa !12
  %156 = getelementptr inbounds nuw i8, ptr %154, i64 8
  %157 = load double, ptr %156, align 8, !tbaa !14
  %158 = getelementptr inbounds nuw i8, ptr %154, i64 16
  %159 = load ptr, ptr %158, align 8, !tbaa !18
  %160 = getelementptr inbounds nuw i8, ptr %154, i64 24
  %161 = load i32, ptr %160, align 8, !tbaa !12
  %162 = icmp ne ptr %159, null
  %163 = zext i1 %162 to i32
  %164 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %155, double noundef %157, i32 noundef %163, i32 noundef %161)
  br label %165

165:                                              ; preds = %10, %152, %133, %113, %95, %78, %61, %44, %27
  %166 = load i8, ptr %13, align 1, !tbaa !6
  %167 = icmp eq i8 %166, 0
  br i1 %167, label %168, label %10, !llvm.loop !20

168:                                              ; preds = %165, %1
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca %struct.DWordS_struct, align 8
  %2 = alloca %struct.LargeS_struct, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  store i64 416611827730, ptr %1, align 8
  tail call void (ptr, ...) @test(ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.10, i32 noundef -123, i32 noundef 97, i32 noundef 123, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, ptr noundef nonnull @.str.11)
  tail call void (ptr, ...) @test(ptr noundef nonnull @.str.12, double noundef 1.000000e+00, double noundef 2.000000e+00, i32 noundef 32764, i64 noundef 12345677823423)
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  store <2 x i32> <i32 21, i32 0>, ptr %2, align 8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store double 2.200000e+01, ptr %3, align 8, !tbaa !14
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr %1, ptr %4, align 8, !tbaa !18
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store <2 x i32> <i32 23, i32 0>, ptr %5, align 8
  call void (ptr, ...) @test(ptr noundef nonnull @.str.13, i64 416611827730, [2 x i64] [i64 19, i64 4626322717216342016], ptr dead_on_return noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 omnipotent char", !11, i64 0}
!11 = !{!"any pointer", !7, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !7, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"double", !7, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"long long", !7, i64 0}
!18 = !{!19, !19, i64 0}
!19 = !{!"p1 _ZTS13DWordS_struct", !11, i64 0}
!20 = distinct !{!20, !21}
!21 = !{!"llvm.loop.mustprogress"}
