; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr90025.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr90025.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@__const.foo.s = private unnamed_addr constant <{ i8, i8, i8, i8, i8, i8, [26 x i8] }> <{ i8 102, i8 111, i8 111, i8 98, i8 97, i8 114, [26 x i8] zeroinitializer }>, align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i8, ptr %0, align 1, !tbaa !6
  %3 = icmp eq i8 %2, 102
  br i1 %3, label %9, label %8

4:                                                ; preds = %25
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 6
  %6 = load i8, ptr %5, align 1, !tbaa !6
  %7 = icmp eq i8 %6, 0
  br i1 %7, label %29, label %129

8:                                                ; preds = %25, %21, %17, %13, %9, %1
  tail call void @abort() #5
  unreachable

9:                                                ; preds = %1
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %11 = load i8, ptr %10, align 1, !tbaa !6
  %12 = icmp eq i8 %11, 111
  br i1 %12, label %13, label %8

13:                                               ; preds = %9
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %15 = load i8, ptr %14, align 1, !tbaa !6
  %16 = icmp eq i8 %15, 111
  br i1 %16, label %17, label %8

17:                                               ; preds = %13
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %19 = load i8, ptr %18, align 1, !tbaa !6
  %20 = icmp eq i8 %19, 98
  br i1 %20, label %21, label %8

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %23 = load i8, ptr %22, align 1, !tbaa !6
  %24 = icmp eq i8 %23, 97
  br i1 %24, label %25, label %8

25:                                               ; preds = %21
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 5
  %27 = load i8, ptr %26, align 1, !tbaa !6
  %28 = icmp eq i8 %27, 114
  br i1 %28, label %4, label %8

29:                                               ; preds = %4
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 7
  %31 = load i8, ptr %30, align 1, !tbaa !6
  %32 = icmp eq i8 %31, 0
  br i1 %32, label %33, label %129

33:                                               ; preds = %29
  %34 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %35 = load i8, ptr %34, align 1, !tbaa !6
  %36 = icmp eq i8 %35, 0
  br i1 %36, label %37, label %129

37:                                               ; preds = %33
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 9
  %39 = load i8, ptr %38, align 1, !tbaa !6
  %40 = icmp eq i8 %39, 0
  br i1 %40, label %41, label %129

41:                                               ; preds = %37
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 10
  %43 = load i8, ptr %42, align 1, !tbaa !6
  %44 = icmp eq i8 %43, 0
  br i1 %44, label %45, label %129

45:                                               ; preds = %41
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 11
  %47 = load i8, ptr %46, align 1, !tbaa !6
  %48 = icmp eq i8 %47, 0
  br i1 %48, label %49, label %129

49:                                               ; preds = %45
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %51 = load i8, ptr %50, align 1, !tbaa !6
  %52 = icmp eq i8 %51, 0
  br i1 %52, label %53, label %129

53:                                               ; preds = %49
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 13
  %55 = load i8, ptr %54, align 1, !tbaa !6
  %56 = icmp eq i8 %55, 0
  br i1 %56, label %57, label %129

57:                                               ; preds = %53
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 14
  %59 = load i8, ptr %58, align 1, !tbaa !6
  %60 = icmp eq i8 %59, 0
  br i1 %60, label %61, label %129

61:                                               ; preds = %57
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 15
  %63 = load i8, ptr %62, align 1, !tbaa !6
  %64 = icmp eq i8 %63, 0
  br i1 %64, label %65, label %129

65:                                               ; preds = %61
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %67 = load i8, ptr %66, align 1, !tbaa !6
  %68 = icmp eq i8 %67, 0
  br i1 %68, label %69, label %129

69:                                               ; preds = %65
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 17
  %71 = load i8, ptr %70, align 1, !tbaa !6
  %72 = icmp eq i8 %71, 0
  br i1 %72, label %73, label %129

73:                                               ; preds = %69
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 18
  %75 = load i8, ptr %74, align 1, !tbaa !6
  %76 = icmp eq i8 %75, 0
  br i1 %76, label %77, label %129

77:                                               ; preds = %73
  %78 = getelementptr inbounds nuw i8, ptr %0, i64 19
  %79 = load i8, ptr %78, align 1, !tbaa !6
  %80 = icmp eq i8 %79, 0
  br i1 %80, label %81, label %129

81:                                               ; preds = %77
  %82 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %83 = load i8, ptr %82, align 1, !tbaa !6
  %84 = icmp eq i8 %83, 0
  br i1 %84, label %85, label %129

85:                                               ; preds = %81
  %86 = getelementptr inbounds nuw i8, ptr %0, i64 21
  %87 = load i8, ptr %86, align 1, !tbaa !6
  %88 = icmp eq i8 %87, 0
  br i1 %88, label %89, label %129

89:                                               ; preds = %85
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 22
  %91 = load i8, ptr %90, align 1, !tbaa !6
  %92 = icmp eq i8 %91, 0
  br i1 %92, label %93, label %129

93:                                               ; preds = %89
  %94 = getelementptr inbounds nuw i8, ptr %0, i64 23
  %95 = load i8, ptr %94, align 1, !tbaa !6
  %96 = icmp eq i8 %95, 0
  br i1 %96, label %97, label %129

97:                                               ; preds = %93
  %98 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %99 = load i8, ptr %98, align 1, !tbaa !6
  %100 = icmp eq i8 %99, 0
  br i1 %100, label %101, label %129

101:                                              ; preds = %97
  %102 = getelementptr inbounds nuw i8, ptr %0, i64 25
  %103 = load i8, ptr %102, align 1, !tbaa !6
  %104 = icmp eq i8 %103, 0
  br i1 %104, label %105, label %129

105:                                              ; preds = %101
  %106 = getelementptr inbounds nuw i8, ptr %0, i64 26
  %107 = load i8, ptr %106, align 1, !tbaa !6
  %108 = icmp eq i8 %107, 0
  br i1 %108, label %109, label %129

109:                                              ; preds = %105
  %110 = getelementptr inbounds nuw i8, ptr %0, i64 27
  %111 = load i8, ptr %110, align 1, !tbaa !6
  %112 = icmp eq i8 %111, 0
  br i1 %112, label %113, label %129

113:                                              ; preds = %109
  %114 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %115 = load i8, ptr %114, align 1, !tbaa !6
  %116 = icmp eq i8 %115, 0
  br i1 %116, label %117, label %129

117:                                              ; preds = %113
  %118 = getelementptr inbounds nuw i8, ptr %0, i64 29
  %119 = load i8, ptr %118, align 1, !tbaa !6
  %120 = icmp eq i8 %119, 0
  br i1 %120, label %121, label %129

121:                                              ; preds = %117
  %122 = getelementptr inbounds nuw i8, ptr %0, i64 30
  %123 = load i8, ptr %122, align 1, !tbaa !6
  %124 = icmp eq i8 %123, 0
  br i1 %124, label %125, label %129

125:                                              ; preds = %121
  %126 = getelementptr inbounds nuw i8, ptr %0, i64 31
  %127 = load i8, ptr %126, align 1, !tbaa !6
  %128 = icmp eq i8 %127, 0
  br i1 %128, label %130, label %129

129:                                              ; preds = %125, %121, %117, %113, %109, %105, %101, %97, %93, %89, %85, %81, %77, %73, %69, %65, %61, %57, %53, %49, %45, %41, %37, %33, %29, %4
  tail call void @abort() #5
  unreachable

130:                                              ; preds = %125
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca [32 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(32) %2, ptr noundef nonnull align 1 dereferenceable(32) @__const.foo.s, i64 32, i1 false)
  %3 = tail call i32 @llvm.bswap.i32(i32 %0)
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i32 %3, ptr %4, align 1, !tbaa !9
  call void @bar(ptr noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.bswap.i32(i32) #4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [32 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(32) %1, ptr noundef nonnull align 1 dereferenceable(32) @__const.foo.s, i64 32, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %2, align 1, !tbaa !9
  call void @bar(ptr noundef nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

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
!10 = !{!"int", !7, i64 0}
