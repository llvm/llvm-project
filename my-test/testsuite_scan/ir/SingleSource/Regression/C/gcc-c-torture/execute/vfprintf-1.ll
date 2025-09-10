; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vfprintf-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vfprintf-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@stdout = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"hello\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.3 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%c\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@.str.7 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

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
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  call void @llvm.va_start.p0(ptr nonnull %3)
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
  %27 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %28 = call i32 @vfprintf(ptr noundef %27, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %4) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #6
  %29 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %30 = call i32 @vfprintf(ptr noundef %29, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %5) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #6
  %31 = icmp eq i32 %30, 5
  br i1 %31, label %104, label %32

32:                                               ; preds = %26
  call void @abort() #7
  unreachable

33:                                               ; preds = %1
  %34 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %35 = call i32 @vfprintf(ptr noundef %34, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %6) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #6
  %36 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %37 = call i32 @vfprintf(ptr noundef %36, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %7) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #6
  %38 = icmp eq i32 %37, 6
  br i1 %38, label %104, label %39

39:                                               ; preds = %33
  call void @abort() #7
  unreachable

40:                                               ; preds = %1
  %41 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %42 = call i32 @vfprintf(ptr noundef %41, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %8) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #6
  %43 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %44 = call i32 @vfprintf(ptr noundef %43, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %9) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #6
  %45 = icmp eq i32 %44, 1
  br i1 %45, label %104, label %46

46:                                               ; preds = %40
  call void @abort() #7
  unreachable

47:                                               ; preds = %1
  %48 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %49 = call i32 @vfprintf(ptr noundef %48, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %10) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #6
  %50 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %51 = call i32 @vfprintf(ptr noundef %50, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %11) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #6
  %52 = icmp eq i32 %51, 0
  br i1 %52, label %104, label %53

53:                                               ; preds = %47
  call void @abort() #7
  unreachable

54:                                               ; preds = %1
  %55 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %56 = call i32 @vfprintf(ptr noundef %55, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %12) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #6
  %57 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %58 = call i32 @vfprintf(ptr noundef %57, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %13) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #6
  %59 = icmp eq i32 %58, 5
  br i1 %59, label %104, label %60

60:                                               ; preds = %54
  call void @abort() #7
  unreachable

61:                                               ; preds = %1
  %62 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %14) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %63 = call i32 @vfprintf(ptr noundef %62, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %14) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %14) #6
  %64 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %65 = call i32 @vfprintf(ptr noundef %64, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %15) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #6
  %66 = icmp eq i32 %65, 6
  br i1 %66, label %104, label %67

67:                                               ; preds = %61
  call void @abort() #7
  unreachable

68:                                               ; preds = %1
  %69 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %16) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %70 = call i32 @vfprintf(ptr noundef %69, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %16) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %16) #6
  %71 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %17) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %72 = call i32 @vfprintf(ptr noundef %71, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %17) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %17) #6
  %73 = icmp eq i32 %72, 1
  br i1 %73, label %104, label %74

74:                                               ; preds = %68
  call void @abort() #7
  unreachable

75:                                               ; preds = %1
  %76 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %18) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %77 = call i32 @vfprintf(ptr noundef %76, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %18) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %18) #6
  %78 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %19) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %79 = call i32 @vfprintf(ptr noundef %78, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %19) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %19) #6
  %80 = icmp eq i32 %79, 0
  br i1 %80, label %104, label %81

81:                                               ; preds = %75
  call void @abort() #7
  unreachable

82:                                               ; preds = %1
  %83 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %20) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %84 = call i32 @vfprintf(ptr noundef %83, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %20) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %20) #6
  %85 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %21) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %86 = call i32 @vfprintf(ptr noundef %85, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %21) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %21) #6
  %87 = icmp eq i32 %86, 1
  br i1 %87, label %104, label %88

88:                                               ; preds = %82
  call void @abort() #7
  unreachable

89:                                               ; preds = %1
  %90 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %22) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %91 = call i32 @vfprintf(ptr noundef %90, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %22) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %22) #6
  %92 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %23) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %93 = call i32 @vfprintf(ptr noundef %92, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %23) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %23) #6
  %94 = icmp eq i32 %93, 7
  br i1 %94, label %104, label %95

95:                                               ; preds = %89
  call void @abort() #7
  unreachable

96:                                               ; preds = %1
  %97 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %24) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !11
  %98 = call i32 @vfprintf(ptr noundef %97, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %24) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %24) #6
  %99 = load ptr, ptr @stdout, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %25) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !11
  %100 = call i32 @vfprintf(ptr noundef %99, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %25) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %25) #6
  %101 = icmp eq i32 %100, 2
  br i1 %101, label %104, label %102

102:                                              ; preds = %96
  call void @abort() #7
  unreachable

103:                                              ; preds = %1
  call void @abort() #7
  unreachable

104:                                              ; preds = %96, %89, %82, %75, %68, %61, %54, %47, %40, %33, %26
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: nofree nounwind
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

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

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i64 0, i64 8, !12, i64 8, i64 8, !12, i64 16, i64 8, !12, i64 24, i64 4, !13, i64 28, i64 4, !13}
!12 = !{!8, !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !9, i64 0}
