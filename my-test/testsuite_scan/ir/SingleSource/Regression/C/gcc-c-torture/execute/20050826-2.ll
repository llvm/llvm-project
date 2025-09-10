; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050826-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050826-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.rtattr = type { i16, i16 }

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local range(i32 -22, 1) i32 @inet_check_attr(ptr readnone captures(none) %0, ptr noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = load ptr, ptr %1, align 8, !tbaa !6
  %4 = icmp eq ptr %3, null
  br i1 %4, label %11, label %5

5:                                                ; preds = %2
  %6 = load i16, ptr %3, align 2, !tbaa !11
  %7 = and i16 %6, -4
  %8 = icmp eq i16 %7, 4
  br i1 %8, label %137, label %9

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store ptr %10, ptr %1, align 8, !tbaa !6
  br label %11

11:                                               ; preds = %9, %2
  %12 = getelementptr i8, ptr %1, i64 8
  %13 = load ptr, ptr %12, align 8, !tbaa !6
  %14 = icmp eq ptr %13, null
  br i1 %14, label %21, label %15

15:                                               ; preds = %11
  %16 = load i16, ptr %13, align 2, !tbaa !11
  %17 = and i16 %16, -4
  %18 = icmp eq i16 %17, 4
  br i1 %18, label %137, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds nuw i8, ptr %13, i64 4
  store ptr %20, ptr %12, align 8, !tbaa !6
  br label %21

21:                                               ; preds = %19, %11
  %22 = getelementptr i8, ptr %1, i64 16
  %23 = load ptr, ptr %22, align 8, !tbaa !6
  %24 = icmp eq ptr %23, null
  br i1 %24, label %31, label %25

25:                                               ; preds = %21
  %26 = load i16, ptr %23, align 2, !tbaa !11
  %27 = and i16 %26, -4
  %28 = icmp eq i16 %27, 4
  br i1 %28, label %137, label %29

29:                                               ; preds = %25
  %30 = getelementptr inbounds nuw i8, ptr %23, i64 4
  store ptr %30, ptr %22, align 8, !tbaa !6
  br label %31

31:                                               ; preds = %29, %21
  %32 = getelementptr i8, ptr %1, i64 24
  %33 = load ptr, ptr %32, align 8, !tbaa !6
  %34 = icmp eq ptr %33, null
  br i1 %34, label %41, label %35

35:                                               ; preds = %31
  %36 = load i16, ptr %33, align 2, !tbaa !11
  %37 = and i16 %36, -4
  %38 = icmp eq i16 %37, 4
  br i1 %38, label %137, label %39

39:                                               ; preds = %35
  %40 = getelementptr inbounds nuw i8, ptr %33, i64 4
  store ptr %40, ptr %32, align 8, !tbaa !6
  br label %41

41:                                               ; preds = %39, %31
  %42 = getelementptr i8, ptr %1, i64 32
  %43 = load ptr, ptr %42, align 8, !tbaa !6
  %44 = icmp eq ptr %43, null
  br i1 %44, label %51, label %45

45:                                               ; preds = %41
  %46 = load i16, ptr %43, align 2, !tbaa !11
  %47 = and i16 %46, -4
  %48 = icmp eq i16 %47, 4
  br i1 %48, label %137, label %49

49:                                               ; preds = %45
  %50 = getelementptr inbounds nuw i8, ptr %43, i64 4
  store ptr %50, ptr %42, align 8, !tbaa !6
  br label %51

51:                                               ; preds = %49, %41
  %52 = getelementptr i8, ptr %1, i64 40
  %53 = load ptr, ptr %52, align 8, !tbaa !6
  %54 = icmp eq ptr %53, null
  br i1 %54, label %61, label %55

55:                                               ; preds = %51
  %56 = load i16, ptr %53, align 2, !tbaa !11
  %57 = and i16 %56, -4
  %58 = icmp eq i16 %57, 4
  br i1 %58, label %137, label %59

59:                                               ; preds = %55
  %60 = getelementptr inbounds nuw i8, ptr %53, i64 4
  store ptr %60, ptr %52, align 8, !tbaa !6
  br label %61

61:                                               ; preds = %59, %51
  %62 = getelementptr i8, ptr %1, i64 48
  %63 = load ptr, ptr %62, align 8, !tbaa !6
  %64 = icmp eq ptr %63, null
  br i1 %64, label %71, label %65

65:                                               ; preds = %61
  %66 = load i16, ptr %63, align 2, !tbaa !11
  %67 = and i16 %66, -4
  %68 = icmp eq i16 %67, 4
  br i1 %68, label %137, label %69

69:                                               ; preds = %65
  %70 = getelementptr inbounds nuw i8, ptr %63, i64 4
  store ptr %70, ptr %62, align 8, !tbaa !6
  br label %71

71:                                               ; preds = %69, %61
  %72 = getelementptr i8, ptr %1, i64 56
  %73 = load ptr, ptr %72, align 8, !tbaa !6
  %74 = icmp eq ptr %73, null
  br i1 %74, label %79, label %75

75:                                               ; preds = %71
  %76 = load i16, ptr %73, align 2, !tbaa !11
  %77 = and i16 %76, -4
  %78 = icmp eq i16 %77, 4
  br i1 %78, label %137, label %79

79:                                               ; preds = %75, %71
  %80 = getelementptr i8, ptr %1, i64 64
  %81 = load ptr, ptr %80, align 8, !tbaa !6
  %82 = icmp eq ptr %81, null
  br i1 %82, label %87, label %83

83:                                               ; preds = %79
  %84 = load i16, ptr %81, align 2, !tbaa !11
  %85 = and i16 %84, -4
  %86 = icmp eq i16 %85, 4
  br i1 %86, label %137, label %87

87:                                               ; preds = %83, %79
  %88 = getelementptr i8, ptr %1, i64 72
  %89 = load ptr, ptr %88, align 8, !tbaa !6
  %90 = icmp eq ptr %89, null
  br i1 %90, label %97, label %91

91:                                               ; preds = %87
  %92 = load i16, ptr %89, align 2, !tbaa !11
  %93 = and i16 %92, -4
  %94 = icmp eq i16 %93, 4
  br i1 %94, label %137, label %95

95:                                               ; preds = %91
  %96 = getelementptr inbounds nuw i8, ptr %89, i64 4
  store ptr %96, ptr %88, align 8, !tbaa !6
  br label %97

97:                                               ; preds = %95, %87
  %98 = getelementptr i8, ptr %1, i64 80
  %99 = load ptr, ptr %98, align 8, !tbaa !6
  %100 = icmp eq ptr %99, null
  br i1 %100, label %107, label %101

101:                                              ; preds = %97
  %102 = load i16, ptr %99, align 2, !tbaa !11
  %103 = and i16 %102, -4
  %104 = icmp eq i16 %103, 4
  br i1 %104, label %137, label %105

105:                                              ; preds = %101
  %106 = getelementptr inbounds nuw i8, ptr %99, i64 4
  store ptr %106, ptr %98, align 8, !tbaa !6
  br label %107

107:                                              ; preds = %105, %97
  %108 = getelementptr i8, ptr %1, i64 88
  %109 = load ptr, ptr %108, align 8, !tbaa !6
  %110 = icmp eq ptr %109, null
  br i1 %110, label %117, label %111

111:                                              ; preds = %107
  %112 = load i16, ptr %109, align 2, !tbaa !11
  %113 = and i16 %112, -4
  %114 = icmp eq i16 %113, 4
  br i1 %114, label %137, label %115

115:                                              ; preds = %111
  %116 = getelementptr inbounds nuw i8, ptr %109, i64 4
  store ptr %116, ptr %108, align 8, !tbaa !6
  br label %117

117:                                              ; preds = %115, %107
  %118 = getelementptr i8, ptr %1, i64 96
  %119 = load ptr, ptr %118, align 8, !tbaa !6
  %120 = icmp eq ptr %119, null
  br i1 %120, label %127, label %121

121:                                              ; preds = %117
  %122 = load i16, ptr %119, align 2, !tbaa !11
  %123 = and i16 %122, -4
  %124 = icmp eq i16 %123, 4
  br i1 %124, label %137, label %125

125:                                              ; preds = %121
  %126 = getelementptr inbounds nuw i8, ptr %119, i64 4
  store ptr %126, ptr %118, align 8, !tbaa !6
  br label %127

127:                                              ; preds = %125, %117
  %128 = getelementptr i8, ptr %1, i64 104
  %129 = load ptr, ptr %128, align 8, !tbaa !6
  %130 = icmp eq ptr %129, null
  br i1 %130, label %137, label %131

131:                                              ; preds = %127
  %132 = load i16, ptr %129, align 2, !tbaa !11
  %133 = and i16 %132, -4
  %134 = icmp eq i16 %133, 4
  br i1 %134, label %137, label %135

135:                                              ; preds = %131
  %136 = getelementptr inbounds nuw i8, ptr %129, i64 4
  store ptr %136, ptr %128, align 8, !tbaa !6
  br label %137

137:                                              ; preds = %127, %135, %131, %121, %111, %101, %91, %83, %75, %65, %55, %45, %35, %25, %15, %5
  %138 = phi i32 [ -22, %5 ], [ -22, %15 ], [ -22, %25 ], [ -22, %35 ], [ -22, %45 ], [ -22, %55 ], [ -22, %65 ], [ -22, %75 ], [ -22, %83 ], [ -22, %91 ], [ -22, %101 ], [ -22, %111 ], [ -22, %121 ], [ -22, %131 ], [ 0, %135 ], [ 0, %127 ]
  ret i32 %138
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [2 x %struct.rtattr], align 4
  %2 = alloca [14 x ptr], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  store i16 12, ptr %1, align 4, !tbaa !11
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 2
  store i16 0, ptr %3, align 2, !tbaa !14
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %5 = load i32, ptr %1, align 4
  store i32 %5, ptr %4, align 4
  store ptr %1, ptr %2, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %1, ptr %6, align 8, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr %1, ptr %7, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store ptr %1, ptr %8, align 8, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 32
  store ptr %1, ptr %9, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 40
  store ptr %1, ptr %10, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 48
  store ptr %1, ptr %11, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 56
  store ptr %1, ptr %12, align 8, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 64
  store ptr %1, ptr %13, align 8, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 72
  store ptr %1, ptr %14, align 8, !tbaa !6
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 80
  store ptr %1, ptr %15, align 8, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 88
  store ptr %1, ptr %16, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %2, i64 96
  store ptr %1, ptr %17, align 8, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %2, i64 104
  store ptr %1, ptr %18, align 8, !tbaa !6
  %19 = call i32 @inet_check_attr(ptr poison, ptr noundef nonnull %2)
  %20 = icmp eq i32 %19, 0
  %21 = trunc i32 %5 to i16
  br i1 %20, label %22, label %49

22:                                               ; preds = %0
  %23 = load <8 x ptr>, ptr %2, align 8
  %24 = insertelement <8 x ptr> poison, ptr %4, i64 0
  %25 = insertelement <8 x ptr> %24, ptr %1, i64 1
  %26 = shufflevector <8 x ptr> %25, <8 x ptr> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1>
  %27 = freeze <8 x ptr> %23
  %28 = icmp eq <8 x ptr> %27, %26
  %29 = load <4 x ptr>, ptr %13, align 8
  %30 = insertelement <4 x ptr> poison, ptr %1, i64 0
  %31 = insertelement <4 x ptr> %30, ptr %4, i64 1
  %32 = shufflevector <4 x ptr> %31, <4 x ptr> poison, <4 x i32> <i32 0, i32 1, i32 1, i32 1>
  %33 = freeze <4 x ptr> %29
  %34 = icmp eq <4 x ptr> %33, %32
  %35 = load ptr, ptr %17, align 8
  %36 = freeze ptr %35
  %37 = icmp eq ptr %36, %4
  %38 = load ptr, ptr %18, align 8
  %39 = icmp eq ptr %38, %4
  %40 = shufflevector <8 x i1> %28, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %41 = select <4 x i1> %40, <4 x i1> %34, <4 x i1> zeroinitializer
  %42 = shufflevector <4 x i1> %41, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %43 = freeze <8 x i1> %42
  %44 = shufflevector <8 x i1> %43, <8 x i1> %28, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
  %45 = bitcast <8 x i1> %44 to i8
  %46 = icmp eq i8 %45, -1
  %47 = and i1 %46, %37
  %48 = select i1 %47, i1 %39, i1 false
  br i1 %48, label %50, label %54

49:                                               ; preds = %0
  call void @abort() #5
  unreachable

50:                                               ; preds = %22
  store ptr %1, ptr %2, align 8, !tbaa !6
  store ptr %1, ptr %7, align 8, !tbaa !6
  store ptr %1, ptr %8, align 8, !tbaa !6
  store ptr %1, ptr %9, align 8, !tbaa !6
  store ptr %1, ptr %11, align 8, !tbaa !6
  store ptr %1, ptr %12, align 8, !tbaa !6
  store ptr %1, ptr %13, align 8, !tbaa !6
  store ptr %1, ptr %14, align 8, !tbaa !6
  store ptr %1, ptr %15, align 8, !tbaa !6
  store ptr %1, ptr %16, align 8, !tbaa !6
  store ptr %1, ptr %17, align 8, !tbaa !6
  store ptr %1, ptr %18, align 8, !tbaa !6
  store ptr null, ptr %6, align 8, !tbaa !6
  %51 = add i16 %21, -8
  store i16 %51, ptr %4, align 4, !tbaa !11
  store ptr %4, ptr %10, align 8, !tbaa !6
  %52 = call i32 @inet_check_attr(ptr poison, ptr noundef nonnull %2)
  %53 = icmp eq i32 %52, -22
  br i1 %53, label %55, label %78

54:                                               ; preds = %22
  call void @abort() #5
  unreachable

55:                                               ; preds = %50
  %56 = load ptr, ptr %6, align 8
  %57 = freeze ptr %56
  %58 = icmp eq ptr %57, null
  %59 = load ptr, ptr %2, align 8, !tbaa !6
  %60 = icmp eq ptr %59, %4
  br i1 %58, label %61, label %80

61:                                               ; preds = %55
  %62 = load <4 x ptr>, ptr %7, align 8
  %63 = insertelement <4 x ptr> poison, ptr %4, i64 0
  %64 = shufflevector <4 x ptr> %63, <4 x ptr> poison, <4 x i32> zeroinitializer
  %65 = freeze <4 x ptr> %62
  %66 = icmp ne <4 x ptr> %65, %64
  %67 = bitcast <4 x i1> %66 to i4
  %68 = icmp eq i4 %67, 0
  %69 = select i1 %60, i1 %68, i1 false
  br i1 %69, label %70, label %81

70:                                               ; preds = %61
  %71 = load <8 x ptr>, ptr %11, align 8
  %72 = insertelement <8 x ptr> poison, ptr %1, i64 0
  %73 = shufflevector <8 x ptr> %72, <8 x ptr> poison, <8 x i32> zeroinitializer
  %74 = freeze <8 x ptr> %71
  %75 = icmp ne <8 x ptr> %74, %73
  %76 = bitcast <8 x i1> %75 to i8
  %77 = icmp eq i8 %76, 0
  br i1 %77, label %83, label %82

78:                                               ; preds = %50
  call void @abort() #5
  unreachable

79:                                               ; preds = %80
  call void @abort() #5
  unreachable

80:                                               ; preds = %55
  br i1 %60, label %79, label %81

81:                                               ; preds = %80, %61
  call void @abort() #5
  unreachable

82:                                               ; preds = %70
  call void @abort() #5
  unreachable

83:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS6rtattr", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !13, i64 0}
!12 = !{!"rtattr", !13, i64 0, !13, i64 2}
!13 = !{!"short", !9, i64 0}
!14 = !{!12, !13, i64 2}
