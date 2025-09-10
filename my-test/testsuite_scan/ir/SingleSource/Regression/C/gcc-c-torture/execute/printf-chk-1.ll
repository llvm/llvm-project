; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/printf-chk-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/printf-chk-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@should_optimize = dso_local global i32 0, align 4
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.3 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%c\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@.str.7 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i32 @__printf_chk(i32 %0, ptr noundef readonly captures(none) %1, ...) local_unnamed_addr #0 {
  %3 = alloca %struct.__va_list, align 8
  %4 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  %5 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %2
  tail call void @abort() #8
  unreachable

8:                                                ; preds = %2
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.va_start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7, !noalias !10
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %4, i64 32, i1 false)
  %9 = load ptr, ptr @stdout, align 8, !tbaa !13, !noalias !10
  %10 = call i32 @vfprintf(ptr noundef %9, ptr noundef %1, ptr dead_on_return noundef nonnull %3) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7, !noalias !10
  call void @llvm.va_end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  ret i32 %10
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %1 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str)
  %2 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  tail call void @abort() #8
  unreachable

5:                                                ; preds = %0
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %6 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str)
  %7 = icmp eq i32 %6, 5
  br i1 %7, label %9, label %8

8:                                                ; preds = %5
  tail call void @abort() #8
  unreachable

9:                                                ; preds = %5
  %10 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %12, label %13

12:                                               ; preds = %9
  tail call void @abort() #8
  unreachable

13:                                               ; preds = %9
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %14 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.1)
  %15 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %17, label %18

17:                                               ; preds = %13
  tail call void @abort() #8
  unreachable

18:                                               ; preds = %13
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %19 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.1)
  %20 = icmp eq i32 %19, 6
  br i1 %20, label %22, label %21

21:                                               ; preds = %18
  tail call void @abort() #8
  unreachable

22:                                               ; preds = %18
  %23 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %25, label %26

25:                                               ; preds = %22
  tail call void @abort() #8
  unreachable

26:                                               ; preds = %22
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %27 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.2)
  %28 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %31

30:                                               ; preds = %26
  tail call void @abort() #8
  unreachable

31:                                               ; preds = %26
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %32 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.2)
  %33 = icmp eq i32 %32, 1
  br i1 %33, label %35, label %34

34:                                               ; preds = %31
  tail call void @abort() #8
  unreachable

35:                                               ; preds = %31
  %36 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %38, label %39

38:                                               ; preds = %35
  tail call void @abort() #8
  unreachable

39:                                               ; preds = %35
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %40 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.3)
  %41 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %43, label %44

43:                                               ; preds = %39
  tail call void @abort() #8
  unreachable

44:                                               ; preds = %39
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %45 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.3)
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %48, label %47

47:                                               ; preds = %44
  tail call void @abort() #8
  unreachable

48:                                               ; preds = %44
  %49 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %52

51:                                               ; preds = %48
  tail call void @abort() #8
  unreachable

52:                                               ; preds = %48
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %53 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str)
  %54 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %56, label %57

56:                                               ; preds = %52
  tail call void @abort() #8
  unreachable

57:                                               ; preds = %52
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %58 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str)
  %59 = icmp eq i32 %58, 5
  br i1 %59, label %61, label %60

60:                                               ; preds = %57
  tail call void @abort() #8
  unreachable

61:                                               ; preds = %57
  %62 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %65

64:                                               ; preds = %61
  tail call void @abort() #8
  unreachable

65:                                               ; preds = %61
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %66 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1)
  %67 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %69, label %70

69:                                               ; preds = %65
  tail call void @abort() #8
  unreachable

70:                                               ; preds = %65
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %71 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1)
  %72 = icmp eq i32 %71, 6
  br i1 %72, label %74, label %73

73:                                               ; preds = %70
  tail call void @abort() #8
  unreachable

74:                                               ; preds = %70
  %75 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %76 = icmp eq i32 %75, 0
  br i1 %76, label %77, label %78

77:                                               ; preds = %74
  tail call void @abort() #8
  unreachable

78:                                               ; preds = %74
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %79 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2)
  %80 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %81 = icmp eq i32 %80, 0
  br i1 %81, label %82, label %83

82:                                               ; preds = %78
  tail call void @abort() #8
  unreachable

83:                                               ; preds = %78
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %84 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2)
  %85 = icmp eq i32 %84, 1
  br i1 %85, label %87, label %86

86:                                               ; preds = %83
  tail call void @abort() #8
  unreachable

87:                                               ; preds = %83
  %88 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %90, label %91

90:                                               ; preds = %87
  tail call void @abort() #8
  unreachable

91:                                               ; preds = %87
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %92 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3)
  %93 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %94 = icmp eq i32 %93, 0
  br i1 %94, label %95, label %96

95:                                               ; preds = %91
  tail call void @abort() #8
  unreachable

96:                                               ; preds = %91
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %97 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3)
  %98 = icmp eq i32 %97, 0
  br i1 %98, label %100, label %99

99:                                               ; preds = %96
  tail call void @abort() #8
  unreachable

100:                                              ; preds = %96
  %101 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %102 = icmp eq i32 %101, 0
  br i1 %102, label %103, label %104

103:                                              ; preds = %100
  tail call void @abort() #8
  unreachable

104:                                              ; preds = %100
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %105 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.5, i32 noundef 120)
  %106 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %108, label %109

108:                                              ; preds = %104
  tail call void @abort() #8
  unreachable

109:                                              ; preds = %104
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %110 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.5, i32 noundef 120)
  %111 = icmp eq i32 %110, 1
  br i1 %111, label %113, label %112

112:                                              ; preds = %109
  tail call void @abort() #8
  unreachable

113:                                              ; preds = %109
  %114 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %115 = icmp eq i32 %114, 0
  br i1 %115, label %116, label %117

116:                                              ; preds = %113
  tail call void @abort() #8
  unreachable

117:                                              ; preds = %113
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %118 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.1)
  %119 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %120 = icmp eq i32 %119, 0
  br i1 %120, label %121, label %122

121:                                              ; preds = %117
  tail call void @abort() #8
  unreachable

122:                                              ; preds = %117
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %123 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.1)
  %124 = icmp eq i32 %123, 7
  br i1 %124, label %126, label %125

125:                                              ; preds = %122
  tail call void @abort() #8
  unreachable

126:                                              ; preds = %122
  %127 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %128 = icmp eq i32 %127, 0
  br i1 %128, label %129, label %130

129:                                              ; preds = %126
  tail call void @abort() #8
  unreachable

130:                                              ; preds = %126
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %131 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.7, i32 noundef 0)
  %132 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %133 = icmp eq i32 %132, 0
  br i1 %133, label %134, label %135

134:                                              ; preds = %130
  tail call void @abort() #8
  unreachable

135:                                              ; preds = %130
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %136 = tail call i32 (i32, ptr, ...) @__printf_chk(i32 poison, ptr noundef nonnull @.str.7, i32 noundef 0)
  %137 = icmp eq i32 %136, 2
  br i1 %137, label %139, label %138

138:                                              ; preds = %135
  tail call void @abort() #8
  unreachable

139:                                              ; preds = %135
  %140 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %141 = icmp eq i32 %140, 0
  br i1 %141, label %142, label %143

142:                                              ; preds = %139
  tail call void @abort() #8
  unreachable

143:                                              ; preds = %139
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #6

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }
attributes #8 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11}
!11 = distinct !{!11, !12, !"vprintf: argument 0"}
!12 = distinct !{!12, !"vprintf"}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 _ZTS8_IO_FILE", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
