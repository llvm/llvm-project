; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/fprintf-chk-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/fprintf-chk-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@should_optimize = dso_local global i32 0, align 4
@stdout = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.3 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%c\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@.str.7 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i32 @__fprintf_chk(ptr noundef captures(none) %0, i32 %1, ptr noundef readonly captures(none) %2, ...) local_unnamed_addr #0 {
  %4 = alloca %struct.__va_list, align 8
  %5 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  %6 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %3
  tail call void @abort() #8
  unreachable

9:                                                ; preds = %3
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.va_start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #7
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %4, i64 32, i1 false), !tbaa.struct !10
  %10 = call i32 @vfprintf(ptr noundef %0, ptr noundef %2, ptr dead_on_return noundef nonnull %5) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #7
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

; Function Attrs: nofree nounwind
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 {
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %1 = load ptr, ptr @stdout, align 8, !tbaa !13
  %2 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %1, i32 poison, ptr noundef nonnull @.str)
  %3 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %0
  tail call void @abort() #8
  unreachable

6:                                                ; preds = %0
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %7 = load ptr, ptr @stdout, align 8, !tbaa !13
  %8 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %7, i32 poison, ptr noundef nonnull @.str)
  %9 = icmp eq i32 %8, 5
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @abort() #8
  unreachable

11:                                               ; preds = %6
  %12 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %15

14:                                               ; preds = %11
  tail call void @abort() #8
  unreachable

15:                                               ; preds = %11
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %16 = load ptr, ptr @stdout, align 8, !tbaa !13
  %17 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %16, i32 poison, ptr noundef nonnull @.str.1)
  %18 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %21

20:                                               ; preds = %15
  tail call void @abort() #8
  unreachable

21:                                               ; preds = %15
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %22 = load ptr, ptr @stdout, align 8, !tbaa !13
  %23 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %22, i32 poison, ptr noundef nonnull @.str.1)
  %24 = icmp eq i32 %23, 6
  br i1 %24, label %26, label %25

25:                                               ; preds = %21
  tail call void @abort() #8
  unreachable

26:                                               ; preds = %21
  %27 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %29, label %30

29:                                               ; preds = %26
  tail call void @abort() #8
  unreachable

30:                                               ; preds = %26
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %31 = load ptr, ptr @stdout, align 8, !tbaa !13
  %32 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %31, i32 poison, ptr noundef nonnull @.str.2)
  %33 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %35, label %36

35:                                               ; preds = %30
  tail call void @abort() #8
  unreachable

36:                                               ; preds = %30
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %37 = load ptr, ptr @stdout, align 8, !tbaa !13
  %38 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %37, i32 poison, ptr noundef nonnull @.str.2)
  %39 = icmp eq i32 %38, 1
  br i1 %39, label %41, label %40

40:                                               ; preds = %36
  tail call void @abort() #8
  unreachable

41:                                               ; preds = %36
  %42 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %43 = icmp eq i32 %42, 0
  br i1 %43, label %44, label %45

44:                                               ; preds = %41
  tail call void @abort() #8
  unreachable

45:                                               ; preds = %41
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %46 = load ptr, ptr @stdout, align 8, !tbaa !13
  %47 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %46, i32 poison, ptr noundef nonnull @.str.3)
  %48 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %50, label %51

50:                                               ; preds = %45
  tail call void @abort() #8
  unreachable

51:                                               ; preds = %45
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %52 = load ptr, ptr @stdout, align 8, !tbaa !13
  %53 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %52, i32 poison, ptr noundef nonnull @.str.3)
  %54 = icmp eq i32 %53, 0
  br i1 %54, label %56, label %55

55:                                               ; preds = %51
  tail call void @abort() #8
  unreachable

56:                                               ; preds = %51
  %57 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %58 = icmp eq i32 %57, 0
  br i1 %58, label %59, label %60

59:                                               ; preds = %56
  tail call void @abort() #8
  unreachable

60:                                               ; preds = %56
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %61 = load ptr, ptr @stdout, align 8, !tbaa !13
  %62 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %61, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str)
  %63 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %66

65:                                               ; preds = %60
  tail call void @abort() #8
  unreachable

66:                                               ; preds = %60
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %67 = load ptr, ptr @stdout, align 8, !tbaa !13
  %68 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %67, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str)
  %69 = icmp eq i32 %68, 5
  br i1 %69, label %71, label %70

70:                                               ; preds = %66
  tail call void @abort() #8
  unreachable

71:                                               ; preds = %66
  %72 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %74, label %75

74:                                               ; preds = %71
  tail call void @abort() #8
  unreachable

75:                                               ; preds = %71
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %76 = load ptr, ptr @stdout, align 8, !tbaa !13
  %77 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %76, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1)
  %78 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %80, label %81

80:                                               ; preds = %75
  tail call void @abort() #8
  unreachable

81:                                               ; preds = %75
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %82 = load ptr, ptr @stdout, align 8, !tbaa !13
  %83 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %82, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1)
  %84 = icmp eq i32 %83, 6
  br i1 %84, label %86, label %85

85:                                               ; preds = %81
  tail call void @abort() #8
  unreachable

86:                                               ; preds = %81
  %87 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %88 = icmp eq i32 %87, 0
  br i1 %88, label %89, label %90

89:                                               ; preds = %86
  tail call void @abort() #8
  unreachable

90:                                               ; preds = %86
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %91 = load ptr, ptr @stdout, align 8, !tbaa !13
  %92 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %91, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2)
  %93 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %94 = icmp eq i32 %93, 0
  br i1 %94, label %95, label %96

95:                                               ; preds = %90
  tail call void @abort() #8
  unreachable

96:                                               ; preds = %90
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %97 = load ptr, ptr @stdout, align 8, !tbaa !13
  %98 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %97, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2)
  %99 = icmp eq i32 %98, 1
  br i1 %99, label %101, label %100

100:                                              ; preds = %96
  tail call void @abort() #8
  unreachable

101:                                              ; preds = %96
  %102 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %104, label %105

104:                                              ; preds = %101
  tail call void @abort() #8
  unreachable

105:                                              ; preds = %101
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %106 = load ptr, ptr @stdout, align 8, !tbaa !13
  %107 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %106, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3)
  %108 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %109 = icmp eq i32 %108, 0
  br i1 %109, label %110, label %111

110:                                              ; preds = %105
  tail call void @abort() #8
  unreachable

111:                                              ; preds = %105
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %112 = load ptr, ptr @stdout, align 8, !tbaa !13
  %113 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %112, i32 poison, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3)
  %114 = icmp eq i32 %113, 0
  br i1 %114, label %116, label %115

115:                                              ; preds = %111
  tail call void @abort() #8
  unreachable

116:                                              ; preds = %111
  %117 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %118 = icmp eq i32 %117, 0
  br i1 %118, label %119, label %120

119:                                              ; preds = %116
  tail call void @abort() #8
  unreachable

120:                                              ; preds = %116
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %121 = load ptr, ptr @stdout, align 8, !tbaa !13
  %122 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %121, i32 poison, ptr noundef nonnull @.str.5, i32 noundef 120)
  %123 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %124 = icmp eq i32 %123, 0
  br i1 %124, label %125, label %126

125:                                              ; preds = %120
  tail call void @abort() #8
  unreachable

126:                                              ; preds = %120
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %127 = load ptr, ptr @stdout, align 8, !tbaa !13
  %128 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %127, i32 poison, ptr noundef nonnull @.str.5, i32 noundef 120)
  %129 = icmp eq i32 %128, 1
  br i1 %129, label %131, label %130

130:                                              ; preds = %126
  tail call void @abort() #8
  unreachable

131:                                              ; preds = %126
  %132 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %133 = icmp eq i32 %132, 0
  br i1 %133, label %134, label %135

134:                                              ; preds = %131
  tail call void @abort() #8
  unreachable

135:                                              ; preds = %131
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %136 = load ptr, ptr @stdout, align 8, !tbaa !13
  %137 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %136, i32 poison, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.1)
  %138 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %139 = icmp eq i32 %138, 0
  br i1 %139, label %140, label %141

140:                                              ; preds = %135
  tail call void @abort() #8
  unreachable

141:                                              ; preds = %135
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %142 = load ptr, ptr @stdout, align 8, !tbaa !13
  %143 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %142, i32 poison, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.1)
  %144 = icmp eq i32 %143, 7
  br i1 %144, label %146, label %145

145:                                              ; preds = %141
  tail call void @abort() #8
  unreachable

146:                                              ; preds = %141
  %147 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %148 = icmp eq i32 %147, 0
  br i1 %148, label %149, label %150

149:                                              ; preds = %146
  tail call void @abort() #8
  unreachable

150:                                              ; preds = %146
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %151 = load ptr, ptr @stdout, align 8, !tbaa !13
  %152 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %151, i32 poison, ptr noundef nonnull @.str.7, i32 noundef 0)
  %153 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %154 = icmp eq i32 %153, 0
  br i1 %154, label %155, label %156

155:                                              ; preds = %150
  tail call void @abort() #8
  unreachable

156:                                              ; preds = %150
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %157 = load ptr, ptr @stdout, align 8, !tbaa !13
  %158 = tail call i32 (ptr, i32, ptr, ...) @__fprintf_chk(ptr noundef %157, i32 poison, ptr noundef nonnull @.str.7, i32 noundef 0)
  %159 = icmp eq i32 %158, 2
  br i1 %159, label %161, label %160

160:                                              ; preds = %156
  tail call void @abort() #8
  unreachable

161:                                              ; preds = %156
  %162 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %163 = icmp eq i32 %162, 0
  br i1 %163, label %164, label %165

164:                                              ; preds = %161
  tail call void @abort() #8
  unreachable

165:                                              ; preds = %161
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!10 = !{i64 0, i64 8, !11, i64 8, i64 8, !11, i64 16, i64 8, !11, i64 24, i64 4, !6, i64 28, i64 4, !6}
!11 = !{!12, !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 _ZTS8_IO_FILE", !12, i64 0}
