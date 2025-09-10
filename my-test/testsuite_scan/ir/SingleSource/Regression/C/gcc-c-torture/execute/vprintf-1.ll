; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vprintf-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vprintf-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.3 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%c\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@.str.7 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @inner(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  %3 = alloca %struct.__va_list, align 8
  %4 = alloca %struct.__va_list, align 8
  %5 = alloca %struct.__va_list, align 8
  %6 = alloca %struct.__va_list, align 8
  %7 = alloca %struct.__va_list, align 8
  %8 = alloca %struct.__va_list, align 8
  %9 = alloca %struct.__va_list, align 8
  %10 = alloca %struct.__va_list, align 8
  %11 = alloca %struct.__va_list, align 8
  %12 = alloca %struct.__va_list, align 8
  %13 = alloca %struct.__va_list, align 8
  %14 = alloca %struct.__va_list, align 8
  %15 = alloca %struct.__va_list, align 8
  %16 = alloca %struct.__va_list, align 8
  %17 = alloca %struct.__va_list, align 8
  %18 = alloca %struct.__va_list, align 8
  %19 = alloca %struct.__va_list, align 8
  %20 = alloca %struct.__va_list, align 8
  %21 = alloca %struct.__va_list, align 8
  %22 = alloca %struct.__va_list, align 8
  %23 = alloca %struct.__va_list, align 8
  %24 = alloca %struct.__va_list, align 8
  %25 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %24) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %25) #6
  call void @llvm.va_start.p0(ptr nonnull %24)
  call void @llvm.va_start.p0(ptr nonnull %25)
  switch i32 %0, label %103 [
    i32 0, label %26
    i32 1, label %33
    i32 2, label %40
    i32 3, label %47
    i32 4, label %54
    i32 5, label %61
    i32 6, label %68
    i32 7, label %75
    i32 8, label %82
    i32 9, label %89
    i32 10, label %96
  ]

26:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %23) #6, !noalias !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %27 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !6
  %28 = call i32 @vfprintf(ptr noundef %27, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %23) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %23) #6, !noalias !6
  call void @llvm.lifetime.start.p0(ptr nonnull %22) #6, !noalias !14
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %29 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !14
  %30 = call i32 @vfprintf(ptr noundef %29, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %22) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %22) #6, !noalias !14
  %31 = icmp eq i32 %30, 5
  br i1 %31, label %104, label %32

32:                                               ; preds = %26
  call void @abort() #7
  unreachable

33:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %21) #6, !noalias !17
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %34 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !17
  %35 = call i32 @vfprintf(ptr noundef %34, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %21) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %21) #6, !noalias !17
  call void @llvm.lifetime.start.p0(ptr nonnull %20) #6, !noalias !20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %36 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !20
  %37 = call i32 @vfprintf(ptr noundef %36, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %20) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %20) #6, !noalias !20
  %38 = icmp eq i32 %37, 6
  br i1 %38, label %104, label %39

39:                                               ; preds = %33
  call void @abort() #7
  unreachable

40:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %19) #6, !noalias !23
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %41 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !23
  %42 = call i32 @vfprintf(ptr noundef %41, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %19) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %19) #6, !noalias !23
  call void @llvm.lifetime.start.p0(ptr nonnull %18) #6, !noalias !26
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %43 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !26
  %44 = call i32 @vfprintf(ptr noundef %43, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %18) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %18) #6, !noalias !26
  %45 = icmp eq i32 %44, 1
  br i1 %45, label %104, label %46

46:                                               ; preds = %40
  call void @abort() #7
  unreachable

47:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %17) #6, !noalias !29
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %48 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !29
  %49 = call i32 @vfprintf(ptr noundef %48, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %17) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %17) #6, !noalias !29
  call void @llvm.lifetime.start.p0(ptr nonnull %16) #6, !noalias !32
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %50 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !32
  %51 = call i32 @vfprintf(ptr noundef %50, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %16) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %16) #6, !noalias !32
  %52 = icmp eq i32 %51, 0
  br i1 %52, label %104, label %53

53:                                               ; preds = %47
  call void @abort() #7
  unreachable

54:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #6, !noalias !35
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %55 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !35
  %56 = call i32 @vfprintf(ptr noundef %55, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %15) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #6, !noalias !35
  call void @llvm.lifetime.start.p0(ptr nonnull %14) #6, !noalias !38
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %57 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !38
  %58 = call i32 @vfprintf(ptr noundef %57, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %14) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %14) #6, !noalias !38
  %59 = icmp eq i32 %58, 5
  br i1 %59, label %104, label %60

60:                                               ; preds = %54
  call void @abort() #7
  unreachable

61:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #6, !noalias !41
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %62 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !41
  %63 = call i32 @vfprintf(ptr noundef %62, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %13) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #6, !noalias !41
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #6, !noalias !44
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %64 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !44
  %65 = call i32 @vfprintf(ptr noundef %64, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %12) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #6, !noalias !44
  %66 = icmp eq i32 %65, 6
  br i1 %66, label %104, label %67

67:                                               ; preds = %61
  call void @abort() #7
  unreachable

68:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #6, !noalias !47
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %69 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !47
  %70 = call i32 @vfprintf(ptr noundef %69, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %11) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #6, !noalias !47
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #6, !noalias !50
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %71 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !50
  %72 = call i32 @vfprintf(ptr noundef %71, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %10) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #6, !noalias !50
  %73 = icmp eq i32 %72, 1
  br i1 %73, label %104, label %74

74:                                               ; preds = %68
  call void @abort() #7
  unreachable

75:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #6, !noalias !53
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %76 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !53
  %77 = call i32 @vfprintf(ptr noundef %76, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %9) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #6, !noalias !53
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #6, !noalias !56
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %78 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !56
  %79 = call i32 @vfprintf(ptr noundef %78, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %8) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #6, !noalias !56
  %80 = icmp eq i32 %79, 0
  br i1 %80, label %104, label %81

81:                                               ; preds = %75
  call void @abort() #7
  unreachable

82:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #6, !noalias !59
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %83 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !59
  %84 = call i32 @vfprintf(ptr noundef %83, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %7) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #6, !noalias !59
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #6, !noalias !62
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %85 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !62
  %86 = call i32 @vfprintf(ptr noundef %85, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %6) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #6, !noalias !62
  %87 = icmp eq i32 %86, 1
  br i1 %87, label %104, label %88

88:                                               ; preds = %82
  call void @abort() #7
  unreachable

89:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #6, !noalias !65
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %90 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !65
  %91 = call i32 @vfprintf(ptr noundef %90, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %5) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #6, !noalias !65
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #6, !noalias !68
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %92 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !68
  %93 = call i32 @vfprintf(ptr noundef %92, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %4) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #6, !noalias !68
  %94 = icmp eq i32 %93, 7
  br i1 %94, label %104, label %95

95:                                               ; preds = %89
  call void @abort() #7
  unreachable

96:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6, !noalias !71
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false)
  %97 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !71
  %98 = call i32 @vfprintf(ptr noundef %97, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %3) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6, !noalias !71
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6, !noalias !74
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %25, i64 32, i1 false)
  %99 = load ptr, ptr @stdout, align 8, !tbaa !9, !noalias !74
  %100 = call i32 @vfprintf(ptr noundef %99, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6, !noalias !74
  %101 = icmp eq i32 %100, 2
  br i1 %101, label %104, label %102

102:                                              ; preds = %96
  call void @abort() #7
  unreachable

103:                                              ; preds = %1
  call void @abort() #7
  unreachable

104:                                              ; preds = %96, %89, %82, %75, %68, %61, %54, %47, %40, %33, %26
  call void @llvm.va_end.p0(ptr nonnull %24)
  call void @llvm.va_end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %25) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %24) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void (i32, ...) @inner(i32 noundef 0)
  tail call void (i32, ...) @inner(i32 noundef 1)
  tail call void (i32, ...) @inner(i32 noundef 2)
  tail call void (i32, ...) @inner(i32 noundef 3)
  tail call void (i32, ...) @inner(i32 noundef 4, ptr noundef nonnull @.str)
  tail call void (i32, ...) @inner(i32 noundef 5, ptr noundef nonnull @.str.1)
  tail call void (i32, ...) @inner(i32 noundef 6, ptr noundef nonnull @.str.2)
  tail call void (i32, ...) @inner(i32 noundef 7, ptr noundef nonnull @.str.3)
  tail call void (i32, ...) @inner(i32 noundef 8, i32 noundef 120)
  tail call void (i32, ...) @inner(i32 noundef 9, ptr noundef nonnull @.str.1)
  tail call void (i32, ...) @inner(i32 noundef 10, i32 noundef 0)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind }
attributes #7 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7}
!7 = distinct !{!7, !8, !"vprintf: argument 0"}
!8 = distinct !{!8, !"vprintf"}
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 _ZTS8_IO_FILE", !11, i64 0}
!11 = !{!"any pointer", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !{!15}
!15 = distinct !{!15, !16, !"vprintf: argument 0"}
!16 = distinct !{!16, !"vprintf"}
!17 = !{!18}
!18 = distinct !{!18, !19, !"vprintf: argument 0"}
!19 = distinct !{!19, !"vprintf"}
!20 = !{!21}
!21 = distinct !{!21, !22, !"vprintf: argument 0"}
!22 = distinct !{!22, !"vprintf"}
!23 = !{!24}
!24 = distinct !{!24, !25, !"vprintf: argument 0"}
!25 = distinct !{!25, !"vprintf"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"vprintf: argument 0"}
!28 = distinct !{!28, !"vprintf"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"vprintf: argument 0"}
!31 = distinct !{!31, !"vprintf"}
!32 = !{!33}
!33 = distinct !{!33, !34, !"vprintf: argument 0"}
!34 = distinct !{!34, !"vprintf"}
!35 = !{!36}
!36 = distinct !{!36, !37, !"vprintf: argument 0"}
!37 = distinct !{!37, !"vprintf"}
!38 = !{!39}
!39 = distinct !{!39, !40, !"vprintf: argument 0"}
!40 = distinct !{!40, !"vprintf"}
!41 = !{!42}
!42 = distinct !{!42, !43, !"vprintf: argument 0"}
!43 = distinct !{!43, !"vprintf"}
!44 = !{!45}
!45 = distinct !{!45, !46, !"vprintf: argument 0"}
!46 = distinct !{!46, !"vprintf"}
!47 = !{!48}
!48 = distinct !{!48, !49, !"vprintf: argument 0"}
!49 = distinct !{!49, !"vprintf"}
!50 = !{!51}
!51 = distinct !{!51, !52, !"vprintf: argument 0"}
!52 = distinct !{!52, !"vprintf"}
!53 = !{!54}
!54 = distinct !{!54, !55, !"vprintf: argument 0"}
!55 = distinct !{!55, !"vprintf"}
!56 = !{!57}
!57 = distinct !{!57, !58, !"vprintf: argument 0"}
!58 = distinct !{!58, !"vprintf"}
!59 = !{!60}
!60 = distinct !{!60, !61, !"vprintf: argument 0"}
!61 = distinct !{!61, !"vprintf"}
!62 = !{!63}
!63 = distinct !{!63, !64, !"vprintf: argument 0"}
!64 = distinct !{!64, !"vprintf"}
!65 = !{!66}
!66 = distinct !{!66, !67, !"vprintf: argument 0"}
!67 = distinct !{!67, !"vprintf"}
!68 = !{!69}
!69 = distinct !{!69, !70, !"vprintf: argument 0"}
!70 = distinct !{!70, !"vprintf"}
!71 = !{!72}
!72 = distinct !{!72, !73, !"vprintf: argument 0"}
!73 = distinct !{!73, !"vprintf"}
!74 = !{!75}
!75 = distinct !{!75, !76, !"vprintf: argument 0"}
!76 = distinct !{!76, !"vprintf"}
