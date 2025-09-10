; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/mempcpy-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/mempcpy-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buf1 = dso_local global [64 x i64] zeroinitializer, align 8
@buf2 = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 256), align 8
@.str = private unnamed_addr constant [10 x i8] c"ABCDEFGHI\00", align 1
@.str.1 = private unnamed_addr constant [11 x i8] c"ABCDEFGHI\00\00", align 1
@.str.2 = private unnamed_addr constant [18 x i8] c"abcdefghijklmnopq\00", align 1
@.str.3 = private unnamed_addr constant [19 x i8] c"abcdefghijklmnopq\00\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"ABCDEF\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"ABCDEFghijklmnopq\00\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"aBCDEFghijklmnopq\00\00", align 1
@.str.9 = private unnamed_addr constant [19 x i8] c"aBcdEFghijklmnopq\00\00", align 1
@buf5 = dso_local local_unnamed_addr global [20 x i64] zeroinitializer, align 8
@.str.10 = private unnamed_addr constant [19 x i8] c"aBcdRSTUVWklmnopq\00\00", align 1
@.str.11 = private unnamed_addr constant [19 x i8] c"aBcdRSTUVWSlmnopq\00\00", align 1
@.str.12 = private unnamed_addr constant [19 x i8] c"aBcdRSTUVWSlmnrsq\00\00", align 1
@.str.13 = private unnamed_addr constant [19 x i8] c"RSTUVWXYVWSlmnrsq\00\00", align 1
@.str.14 = private unnamed_addr constant [19 x i8] c"RSTUVWXYZ01234567\00\00", align 1
@.str.15 = private unnamed_addr constant [19 x i8] c"aBcdRSTUVWkSmnopq\00\00", align 1
@.str.16 = private unnamed_addr constant [19 x i8] c"aBcdRSTUVWkSmnrsq\00\00", align 1
@buf7 = dso_local local_unnamed_addr global [20 x i8] zeroinitializer, align 1
@inside_main = external local_unnamed_addr global i32, align 4
@.str.17 = private unnamed_addr constant [20 x i8] c"RSTUVWXYZ0123456789\00", align 1
@.str.18 = private unnamed_addr constant [10 x i8] c"rstuvwxyz\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @test(ptr noundef writeonly captures(address) %0, ptr noundef writeonly captures(address) %1, ptr noundef readonly captures(none) %2, i32 noundef %3) local_unnamed_addr #0 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(9) @buf1, ptr noundef nonnull align 1 dereferenceable(9) @.str, i64 9, i1 false)
  %5 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(11) @buf1, ptr noundef nonnull dereferenceable(11) @.str.1, i64 11)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %4
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(17) @buf1, ptr noundef nonnull align 1 dereferenceable(17) @.str.2, i64 17, i1 false)
  %9 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.3, i64 19)
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #5
  unreachable

12:                                               ; preds = %8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(6) %0, ptr noundef nonnull align 1 dereferenceable(6) @.str.4, i64 6, i1 false)
  %13 = icmp eq ptr %0, @buf1
  br i1 %13, label %14, label %17

14:                                               ; preds = %12
  %15 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.5, i64 19)
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %18, label %17

17:                                               ; preds = %14, %12
  tail call void @abort() #5
  unreachable

18:                                               ; preds = %14
  store i8 97, ptr @buf1, align 8
  %19 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.7, i64 19)
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %22, label %21

21:                                               ; preds = %18
  tail call void @abort() #5
  unreachable

22:                                               ; preds = %18
  store i16 25699, ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 2), align 2
  %23 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.9, i64 19)
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @abort() #5
  unreachable

26:                                               ; preds = %22
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(6) getelementptr inbounds nuw (i8, ptr @buf1, i64 4), ptr noundef nonnull align 8 dereferenceable(6) @buf5, i64 6, i1 false)
  %27 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.10, i64 19)
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %30, label %29

29:                                               ; preds = %26
  tail call void @abort() #5
  unreachable

30:                                               ; preds = %26
  %31 = load i8, ptr getelementptr inbounds nuw (i8, ptr @buf5, i64 1), align 1
  store i8 %31, ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 10), align 2
  %32 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.11, i64 19)
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %35, label %34

34:                                               ; preds = %30
  tail call void @abort() #5
  unreachable

35:                                               ; preds = %30
  %36 = load i16, ptr %2, align 1
  store i16 %36, ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 14), align 2
  %37 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.12, i64 19)
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %35
  tail call void @abort() #5
  unreachable

40:                                               ; preds = %35
  %41 = load i64, ptr @buf5, align 8
  store i64 %41, ptr @buf1, align 8
  %42 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.13, i64 19)
  %43 = icmp eq i32 %42, 0
  br i1 %43, label %45, label %44

44:                                               ; preds = %40
  tail call void @abort() #5
  unreachable

45:                                               ; preds = %40
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(17) @buf1, ptr noundef nonnull align 8 dereferenceable(17) @buf5, i64 17, i1 false)
  %46 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.14, i64 19)
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %49, label %48

48:                                               ; preds = %45
  tail call void @abort() #5
  unreachable

49:                                               ; preds = %45
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(19) @buf1, ptr noundef nonnull align 1 dereferenceable(19) @.str.9, i64 19, i1 false)
  %50 = add nsw i32 %3, 6
  %51 = sext i32 %50 to i64
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 getelementptr inbounds nuw (i8, ptr @buf1, i64 4), ptr nonnull align 8 @buf5, i64 %51, i1 false)
  %52 = icmp eq i32 %3, 0
  br i1 %52, label %53, label %56

53:                                               ; preds = %49
  %54 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.10, i64 19)
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %57, label %56

56:                                               ; preds = %53, %49
  tail call void @abort() #5
  unreachable

57:                                               ; preds = %53
  %58 = load i8, ptr getelementptr inbounds nuw (i8, ptr @buf5, i64 1), align 1
  store i8 %58, ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 11), align 1
  %59 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.15, i64 19)
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %62, label %61

61:                                               ; preds = %57
  tail call void @abort() #5
  unreachable

62:                                               ; preds = %57
  %63 = load i16, ptr %2, align 1
  store i16 %63, ptr getelementptr inbounds nuw (i8, ptr @buf1, i64 14), align 2
  %64 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) @buf1, ptr noundef nonnull dereferenceable(19) @.str.16, i64 19)
  %65 = icmp eq i32 %64, 0
  br i1 %65, label %67, label %66

66:                                               ; preds = %62
  tail call void @abort() #5
  unreachable

67:                                               ; preds = %62
  %68 = load ptr, ptr @buf2, align 8, !tbaa !6
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %68, ptr noundef nonnull align 1 dereferenceable(9) @.str, i64 9, i1 false)
  %69 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(11) %68, ptr noundef nonnull dereferenceable(11) @.str.1, i64 11)
  %70 = icmp eq i32 %69, 0
  br i1 %70, label %72, label %71

71:                                               ; preds = %67
  tail call void @abort() #5
  unreachable

72:                                               ; preds = %67
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(17) %68, ptr noundef nonnull align 1 dereferenceable(17) @.str.2, i64 17, i1 false)
  %73 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %68, ptr noundef nonnull dereferenceable(19) @.str.3, i64 19)
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %76, label %75

75:                                               ; preds = %72
  tail call void @abort() #5
  unreachable

76:                                               ; preds = %72
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %1, ptr noundef nonnull align 1 dereferenceable(6) @.str.4, i64 6, i1 false)
  %77 = load ptr, ptr @buf2, align 8, !tbaa !6
  %78 = icmp eq ptr %1, %77
  br i1 %78, label %79, label %82

79:                                               ; preds = %76
  %80 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %77, ptr noundef nonnull dereferenceable(19) @.str.5, i64 19)
  %81 = icmp eq i32 %80, 0
  br i1 %81, label %83, label %82

82:                                               ; preds = %79, %76
  tail call void @abort() #5
  unreachable

83:                                               ; preds = %79
  store i8 97, ptr %1, align 1
  %84 = load ptr, ptr @buf2, align 8, !tbaa !6
  %85 = icmp eq ptr %1, %84
  br i1 %85, label %86, label %89

86:                                               ; preds = %83
  %87 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %84, ptr noundef nonnull dereferenceable(19) @.str.7, i64 19)
  %88 = icmp eq i32 %87, 0
  br i1 %88, label %90, label %89

89:                                               ; preds = %86, %83
  tail call void @abort() #5
  unreachable

90:                                               ; preds = %86
  %91 = getelementptr inbounds nuw i8, ptr %1, i64 2
  store i16 25699, ptr %91, align 1
  %92 = load ptr, ptr @buf2, align 8, !tbaa !6
  %93 = icmp eq ptr %1, %92
  br i1 %93, label %94, label %97

94:                                               ; preds = %90
  %95 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %92, ptr noundef nonnull dereferenceable(19) @.str.9, i64 19)
  %96 = icmp eq i32 %95, 0
  br i1 %96, label %98, label %97

97:                                               ; preds = %94, %90
  tail call void @abort() #5
  unreachable

98:                                               ; preds = %94
  %99 = getelementptr inbounds nuw i8, ptr %1, i64 4
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %99, ptr noundef nonnull align 1 dereferenceable(6) @buf7, i64 6, i1 false)
  %100 = load ptr, ptr @buf2, align 8, !tbaa !6
  %101 = icmp eq ptr %1, %100
  br i1 %101, label %102, label %105

102:                                              ; preds = %98
  %103 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %100, ptr noundef nonnull dereferenceable(19) @.str.10, i64 19)
  %104 = icmp eq i32 %103, 0
  br i1 %104, label %106, label %105

105:                                              ; preds = %102, %98
  tail call void @abort() #5
  unreachable

106:                                              ; preds = %102
  %107 = getelementptr inbounds nuw i8, ptr %100, i64 10
  %108 = load i8, ptr getelementptr inbounds nuw (i8, ptr @buf7, i64 1), align 1
  store i8 %108, ptr %107, align 1
  %109 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %100, ptr noundef nonnull dereferenceable(19) @.str.11, i64 19)
  %110 = icmp eq i32 %109, 0
  br i1 %110, label %112, label %111

111:                                              ; preds = %106
  tail call void @abort() #5
  unreachable

112:                                              ; preds = %106
  %113 = getelementptr inbounds nuw i8, ptr %1, i64 14
  %114 = load i16, ptr %2, align 1
  store i16 %114, ptr %113, align 1
  %115 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %100, ptr noundef nonnull dereferenceable(19) @.str.12, i64 19)
  %116 = icmp eq i32 %115, 0
  br i1 %116, label %118, label %117

117:                                              ; preds = %112
  tail call void @abort() #5
  unreachable

118:                                              ; preds = %112
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(19) %1, ptr noundef nonnull align 1 dereferenceable(19) @.str.9, i64 19, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %99, ptr nonnull align 1 @buf7, i64 %51, i1 false)
  %119 = getelementptr inbounds nuw i8, ptr %99, i64 %51
  %120 = load ptr, ptr @buf2, align 8, !tbaa !6
  %121 = getelementptr inbounds nuw i8, ptr %120, i64 10
  %122 = icmp eq ptr %119, %121
  br i1 %122, label %123, label %126

123:                                              ; preds = %118
  %124 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %120, ptr noundef nonnull dereferenceable(19) @.str.10, i64 19)
  %125 = icmp eq i32 %124, 0
  br i1 %125, label %127, label %126

126:                                              ; preds = %123, %118
  tail call void @abort() #5
  unreachable

127:                                              ; preds = %123
  %128 = getelementptr inbounds nuw i8, ptr %120, i64 11
  %129 = load i8, ptr getelementptr inbounds nuw (i8, ptr @buf7, i64 1), align 1
  store i8 %129, ptr %128, align 1
  %130 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %120, ptr noundef nonnull dereferenceable(19) @.str.15, i64 19)
  %131 = icmp eq i32 %130, 0
  br i1 %131, label %133, label %132

132:                                              ; preds = %127
  tail call void @abort() #5
  unreachable

133:                                              ; preds = %127
  %134 = load i16, ptr %2, align 1
  store i16 %134, ptr %113, align 1
  %135 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(19) %120, ptr noundef nonnull dereferenceable(19) @.str.16, i64 19)
  %136 = icmp eq i32 %135, 0
  br i1 %136, label %138, label %137

137:                                              ; preds = %133
  tail call void @abort() #5
  unreachable

138:                                              ; preds = %133
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #3 {
  store i32 0, ptr @inside_main, align 4, !tbaa !11
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(20) @buf5, ptr noundef nonnull align 1 dereferenceable(20) @.str.17, i64 20, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(20) @buf7, ptr noundef nonnull align 1 dereferenceable(20) @.str.17, i64 20, i1 false)
  %1 = load ptr, ptr @buf2, align 8, !tbaa !6
  tail call void @test(ptr noundef nonnull @buf1, ptr noundef %1, ptr noundef nonnull @.str.18, i32 noundef 0)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #4

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: read) }
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
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
