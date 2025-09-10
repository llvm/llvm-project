; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vfprintf-chk-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vfprintf-chk-1.c"
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
define dso_local noundef i32 @__vfprintf_chk(ptr noundef captures(none) %0, i32 %1, ptr noundef readonly captures(none) %2, ptr dead_on_return noundef readonly captures(none) %3) local_unnamed_addr #0 {
  %5 = alloca %struct.__va_list, align 8
  %6 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %4
  tail call void @abort() #7
  unreachable

9:                                                ; preds = %4
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %10 = call i32 @vfprintf(ptr noundef %0, ptr noundef %2, ptr dead_on_return noundef nonnull %5) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #8
  ret i32 %10
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @inner(i32 noundef %0, ...) local_unnamed_addr #5 {
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
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  call void @llvm.va_start.p0(ptr nonnull %3)
  switch i32 %0, label %191 [
    i32 0, label %26
    i32 1, label %41
    i32 2, label %56
    i32 3, label %71
    i32 4, label %86
    i32 5, label %101
    i32 6, label %116
    i32 7, label %131
    i32 8, label %146
    i32 9, label %161
    i32 10, label %176
  ]

26:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %27 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %28 = call i32 @__vfprintf_chk(ptr noundef %27, i32 poison, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #8
  %29 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %32

31:                                               ; preds = %26
  call void @abort() #7
  unreachable

32:                                               ; preds = %26
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %33 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %34 = call i32 @__vfprintf_chk(ptr noundef %33, i32 poison, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #8
  %35 = icmp eq i32 %34, 5
  br i1 %35, label %37, label %36

36:                                               ; preds = %32
  call void @abort() #7
  unreachable

37:                                               ; preds = %32
  %38 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %192

40:                                               ; preds = %37
  call void @abort() #7
  unreachable

41:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %42 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %43 = call i32 @__vfprintf_chk(ptr noundef %42, i32 poison, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #8
  %44 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %45 = icmp eq i32 %44, 0
  br i1 %45, label %46, label %47

46:                                               ; preds = %41
  call void @abort() #7
  unreachable

47:                                               ; preds = %41
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %48 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %49 = call i32 @__vfprintf_chk(ptr noundef %48, i32 poison, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #8
  %50 = icmp eq i32 %49, 6
  br i1 %50, label %52, label %51

51:                                               ; preds = %47
  call void @abort() #7
  unreachable

52:                                               ; preds = %47
  %53 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %54 = icmp eq i32 %53, 0
  br i1 %54, label %55, label %192

55:                                               ; preds = %52
  call void @abort() #7
  unreachable

56:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %57 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %58 = call i32 @__vfprintf_chk(ptr noundef %57, i32 poison, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #8
  %59 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %61, label %62

61:                                               ; preds = %56
  call void @abort() #7
  unreachable

62:                                               ; preds = %56
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %63 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %64 = call i32 @__vfprintf_chk(ptr noundef %63, i32 poison, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #8
  %65 = icmp eq i32 %64, 1
  br i1 %65, label %67, label %66

66:                                               ; preds = %62
  call void @abort() #7
  unreachable

67:                                               ; preds = %62
  %68 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %192

70:                                               ; preds = %67
  call void @abort() #7
  unreachable

71:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  %72 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %73 = call i32 @__vfprintf_chk(ptr noundef %72, i32 poison, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #8
  %74 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %75 = icmp eq i32 %74, 0
  br i1 %75, label %76, label %77

76:                                               ; preds = %71
  call void @abort() #7
  unreachable

77:                                               ; preds = %71
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %78 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %79 = call i32 @__vfprintf_chk(ptr noundef %78, i32 poison, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #8
  %80 = icmp eq i32 %79, 0
  br i1 %80, label %82, label %81

81:                                               ; preds = %77
  call void @abort() #7
  unreachable

82:                                               ; preds = %77
  %83 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %84 = icmp eq i32 %83, 0
  br i1 %84, label %85, label %192

85:                                               ; preds = %82
  call void @abort() #7
  unreachable

86:                                               ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %87 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %88 = call i32 @__vfprintf_chk(ptr noundef %87, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #8
  %89 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %90 = icmp eq i32 %89, 0
  br i1 %90, label %91, label %92

91:                                               ; preds = %86
  call void @abort() #7
  unreachable

92:                                               ; preds = %86
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %93 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %94 = call i32 @__vfprintf_chk(ptr noundef %93, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #8
  %95 = icmp eq i32 %94, 5
  br i1 %95, label %97, label %96

96:                                               ; preds = %92
  call void @abort() #7
  unreachable

97:                                               ; preds = %92
  %98 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %99 = icmp eq i32 %98, 0
  br i1 %99, label %100, label %192

100:                                              ; preds = %97
  call void @abort() #7
  unreachable

101:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %102 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %14) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %103 = call i32 @__vfprintf_chk(ptr noundef %102, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %14) #8
  %104 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %105 = icmp eq i32 %104, 0
  br i1 %105, label %106, label %107

106:                                              ; preds = %101
  call void @abort() #7
  unreachable

107:                                              ; preds = %101
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %108 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %109 = call i32 @__vfprintf_chk(ptr noundef %108, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #8
  %110 = icmp eq i32 %109, 6
  br i1 %110, label %112, label %111

111:                                              ; preds = %107
  call void @abort() #7
  unreachable

112:                                              ; preds = %107
  %113 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %114 = icmp eq i32 %113, 0
  br i1 %114, label %115, label %192

115:                                              ; preds = %112
  call void @abort() #7
  unreachable

116:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %117 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %16) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %118 = call i32 @__vfprintf_chk(ptr noundef %117, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %16) #8
  %119 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %120 = icmp eq i32 %119, 0
  br i1 %120, label %121, label %122

121:                                              ; preds = %116
  call void @abort() #7
  unreachable

122:                                              ; preds = %116
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %123 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %17) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %124 = call i32 @__vfprintf_chk(ptr noundef %123, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %17) #8
  %125 = icmp eq i32 %124, 1
  br i1 %125, label %127, label %126

126:                                              ; preds = %122
  call void @abort() #7
  unreachable

127:                                              ; preds = %122
  %128 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %129 = icmp eq i32 %128, 0
  br i1 %129, label %130, label %192

130:                                              ; preds = %127
  call void @abort() #7
  unreachable

131:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %132 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %18) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %133 = call i32 @__vfprintf_chk(ptr noundef %132, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %18)
  call void @llvm.lifetime.end.p0(ptr nonnull %18) #8
  %134 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %135 = icmp eq i32 %134, 0
  br i1 %135, label %136, label %137

136:                                              ; preds = %131
  call void @abort() #7
  unreachable

137:                                              ; preds = %131
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %138 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %19) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %139 = call i32 @__vfprintf_chk(ptr noundef %138, i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %19) #8
  %140 = icmp eq i32 %139, 0
  br i1 %140, label %142, label %141

141:                                              ; preds = %137
  call void @abort() #7
  unreachable

142:                                              ; preds = %137
  %143 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %144 = icmp eq i32 %143, 0
  br i1 %144, label %145, label %192

145:                                              ; preds = %142
  call void @abort() #7
  unreachable

146:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %147 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %20) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %148 = call i32 @__vfprintf_chk(ptr noundef %147, i32 poison, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %20) #8
  %149 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %150 = icmp eq i32 %149, 0
  br i1 %150, label %151, label %152

151:                                              ; preds = %146
  call void @abort() #7
  unreachable

152:                                              ; preds = %146
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %153 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %21) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %154 = call i32 @__vfprintf_chk(ptr noundef %153, i32 poison, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %21)
  call void @llvm.lifetime.end.p0(ptr nonnull %21) #8
  %155 = icmp eq i32 %154, 1
  br i1 %155, label %157, label %156

156:                                              ; preds = %152
  call void @abort() #7
  unreachable

157:                                              ; preds = %152
  %158 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %159 = icmp eq i32 %158, 0
  br i1 %159, label %160, label %192

160:                                              ; preds = %157
  call void @abort() #7
  unreachable

161:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %162 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %22) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %163 = call i32 @__vfprintf_chk(ptr noundef %162, i32 poison, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %22) #8
  %164 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %165 = icmp eq i32 %164, 0
  br i1 %165, label %166, label %167

166:                                              ; preds = %161
  call void @abort() #7
  unreachable

167:                                              ; preds = %161
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %168 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %23) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %169 = call i32 @__vfprintf_chk(ptr noundef %168, i32 poison, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %23) #8
  %170 = icmp eq i32 %169, 7
  br i1 %170, label %172, label %171

171:                                              ; preds = %167
  call void @abort() #7
  unreachable

172:                                              ; preds = %167
  %173 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %174 = icmp eq i32 %173, 0
  br i1 %174, label %175, label %192

175:                                              ; preds = %172
  call void @abort() #7
  unreachable

176:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %177 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %24) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !10
  %178 = call i32 @__vfprintf_chk(ptr noundef %177, i32 poison, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %24)
  call void @llvm.lifetime.end.p0(ptr nonnull %24) #8
  %179 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %180 = icmp eq i32 %179, 0
  br i1 %180, label %181, label %182

181:                                              ; preds = %176
  call void @abort() #7
  unreachable

182:                                              ; preds = %176
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  %183 = load ptr, ptr @stdout, align 8, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %25) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !10
  %184 = call i32 @__vfprintf_chk(ptr noundef %183, i32 poison, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %25) #8
  %185 = icmp eq i32 %184, 2
  br i1 %185, label %187, label %186

186:                                              ; preds = %182
  call void @abort() #7
  unreachable

187:                                              ; preds = %182
  %188 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %189 = icmp eq i32 %188, 0
  br i1 %189, label %190, label %192

190:                                              ; preds = %187
  call void @abort() #7
  unreachable

191:                                              ; preds = %1
  call void @abort() #7
  unreachable

192:                                              ; preds = %187, %172, %157, %142, %127, %112, %97, %82, %67, %52, %37
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #6

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
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

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #7 = { cold noreturn nounwind }
attributes #8 = { nounwind }

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
