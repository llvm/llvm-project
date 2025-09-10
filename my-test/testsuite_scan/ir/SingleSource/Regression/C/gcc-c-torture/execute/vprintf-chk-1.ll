; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vprintf-chk-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/vprintf-chk-1.c"
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
define dso_local noundef i32 @__vprintf_chk(i32 %0, ptr noundef readonly captures(none) %1, ptr dead_on_return noundef readonly captures(none) %2) local_unnamed_addr #0 {
  %4 = alloca %struct.__va_list, align 8
  %5 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #7
  unreachable

8:                                                ; preds = %3
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #8, !noalias !10
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false)
  %9 = load ptr, ptr @stdout, align 8, !tbaa !13, !noalias !10
  %10 = call i32 @vfprintf(ptr noundef %9, ptr noundef %1, ptr dead_on_return noundef nonnull %4) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #8, !noalias !10
  ret i32 %10
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind uwtable
define dso_local void @inner(i32 noundef %0, ...) local_unnamed_addr #4 {
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
  switch i32 %0, label %169 [
    i32 0, label %26
    i32 1, label %39
    i32 2, label %52
    i32 3, label %65
    i32 4, label %78
    i32 5, label %91
    i32 6, label %104
    i32 7, label %117
    i32 8, label %130
    i32 9, label %143
    i32 10, label %156
  ]

26:                                               ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %27 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #8
  %28 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %31

30:                                               ; preds = %26
  call void @abort() #7
  unreachable

31:                                               ; preds = %26
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %32 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str, ptr dead_on_return noundef nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #8
  %33 = icmp eq i32 %32, 5
  br i1 %33, label %35, label %34

34:                                               ; preds = %31
  call void @abort() #7
  unreachable

35:                                               ; preds = %31
  %36 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %38, label %170

38:                                               ; preds = %35
  call void @abort() #7
  unreachable

39:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %40 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #8
  %41 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %43, label %44

43:                                               ; preds = %39
  call void @abort() #7
  unreachable

44:                                               ; preds = %39
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %45 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.1, ptr dead_on_return noundef nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #8
  %46 = icmp eq i32 %45, 6
  br i1 %46, label %48, label %47

47:                                               ; preds = %44
  call void @abort() #7
  unreachable

48:                                               ; preds = %44
  %49 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %170

51:                                               ; preds = %48
  call void @abort() #7
  unreachable

52:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %53 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #8
  %54 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %56, label %57

56:                                               ; preds = %52
  call void @abort() #7
  unreachable

57:                                               ; preds = %52
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %58 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.2, ptr dead_on_return noundef nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #8
  %59 = icmp eq i32 %58, 1
  br i1 %59, label %61, label %60

60:                                               ; preds = %57
  call void @abort() #7
  unreachable

61:                                               ; preds = %57
  %62 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %170

64:                                               ; preds = %61
  call void @abort() #7
  unreachable

65:                                               ; preds = %1
  store volatile i32 1, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %66 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #8
  %67 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %69, label %70

69:                                               ; preds = %65
  call void @abort() #7
  unreachable

70:                                               ; preds = %65
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %71 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.3, ptr dead_on_return noundef nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #8
  %72 = icmp eq i32 %71, 0
  br i1 %72, label %74, label %73

73:                                               ; preds = %70
  call void @abort() #7
  unreachable

74:                                               ; preds = %70
  %75 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %76 = icmp eq i32 %75, 0
  br i1 %76, label %77, label %170

77:                                               ; preds = %74
  call void @abort() #7
  unreachable

78:                                               ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %79 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #8
  %80 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %81 = icmp eq i32 %80, 0
  br i1 %81, label %82, label %83

82:                                               ; preds = %78
  call void @abort() #7
  unreachable

83:                                               ; preds = %78
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %84 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #8
  %85 = icmp eq i32 %84, 5
  br i1 %85, label %87, label %86

86:                                               ; preds = %83
  call void @abort() #7
  unreachable

87:                                               ; preds = %83
  %88 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %90, label %170

90:                                               ; preds = %87
  call void @abort() #7
  unreachable

91:                                               ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %14) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %92 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %14) #8
  %93 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %94 = icmp eq i32 %93, 0
  br i1 %94, label %95, label %96

95:                                               ; preds = %91
  call void @abort() #7
  unreachable

96:                                               ; preds = %91
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %97 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #8
  %98 = icmp eq i32 %97, 6
  br i1 %98, label %100, label %99

99:                                               ; preds = %96
  call void @abort() #7
  unreachable

100:                                              ; preds = %96
  %101 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %102 = icmp eq i32 %101, 0
  br i1 %102, label %103, label %170

103:                                              ; preds = %100
  call void @abort() #7
  unreachable

104:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %16) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %105 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %16) #8
  %106 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %108, label %109

108:                                              ; preds = %104
  call void @abort() #7
  unreachable

109:                                              ; preds = %104
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %17) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %110 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %17) #8
  %111 = icmp eq i32 %110, 1
  br i1 %111, label %113, label %112

112:                                              ; preds = %109
  call void @abort() #7
  unreachable

113:                                              ; preds = %109
  %114 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %115 = icmp eq i32 %114, 0
  br i1 %115, label %116, label %170

116:                                              ; preds = %113
  call void @abort() #7
  unreachable

117:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %18) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %118 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %18)
  call void @llvm.lifetime.end.p0(ptr nonnull %18) #8
  %119 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %120 = icmp eq i32 %119, 0
  br i1 %120, label %121, label %122

121:                                              ; preds = %117
  call void @abort() #7
  unreachable

122:                                              ; preds = %117
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %19) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %123 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.4, ptr dead_on_return noundef nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %19) #8
  %124 = icmp eq i32 %123, 0
  br i1 %124, label %126, label %125

125:                                              ; preds = %122
  call void @abort() #7
  unreachable

126:                                              ; preds = %122
  %127 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %128 = icmp eq i32 %127, 0
  br i1 %128, label %129, label %170

129:                                              ; preds = %126
  call void @abort() #7
  unreachable

130:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %20) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %131 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %20) #8
  %132 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %133 = icmp eq i32 %132, 0
  br i1 %133, label %134, label %135

134:                                              ; preds = %130
  call void @abort() #7
  unreachable

135:                                              ; preds = %130
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %21) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %136 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.5, ptr dead_on_return noundef nonnull %21)
  call void @llvm.lifetime.end.p0(ptr nonnull %21) #8
  %137 = icmp eq i32 %136, 1
  br i1 %137, label %139, label %138

138:                                              ; preds = %135
  call void @abort() #7
  unreachable

139:                                              ; preds = %135
  %140 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %141 = icmp eq i32 %140, 0
  br i1 %141, label %142, label %170

142:                                              ; preds = %139
  call void @abort() #7
  unreachable

143:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %22) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %144 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %22) #8
  %145 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %146 = icmp eq i32 %145, 0
  br i1 %146, label %147, label %148

147:                                              ; preds = %143
  call void @abort() #7
  unreachable

148:                                              ; preds = %143
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %23) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %149 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.6, ptr dead_on_return noundef nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %23) #8
  %150 = icmp eq i32 %149, 7
  br i1 %150, label %152, label %151

151:                                              ; preds = %148
  call void @abort() #7
  unreachable

152:                                              ; preds = %148
  %153 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %154 = icmp eq i32 %153, 0
  br i1 %154, label %155, label %170

155:                                              ; preds = %152
  call void @abort() #7
  unreachable

156:                                              ; preds = %1
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %24) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %2, i64 32, i1 false), !tbaa.struct !16
  %157 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %24)
  call void @llvm.lifetime.end.p0(ptr nonnull %24) #8
  %158 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %159 = icmp eq i32 %158, 0
  br i1 %159, label %160, label %161

160:                                              ; preds = %156
  call void @abort() #7
  unreachable

161:                                              ; preds = %156
  store volatile i32 0, ptr @should_optimize, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %25) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %3, i64 32, i1 false), !tbaa.struct !16
  %162 = call i32 @__vprintf_chk(i32 poison, ptr noundef nonnull @.str.7, ptr dead_on_return noundef nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %25) #8
  %163 = icmp eq i32 %162, 2
  br i1 %163, label %165, label %164

164:                                              ; preds = %161
  call void @abort() #7
  unreachable

165:                                              ; preds = %161
  %166 = load volatile i32, ptr @should_optimize, align 4, !tbaa !6
  %167 = icmp eq i32 %166, 0
  br i1 %167, label %168, label %170

168:                                              ; preds = %165
  call void @abort() #7
  unreachable

169:                                              ; preds = %1
  call void @abort() #7
  unreachable

170:                                              ; preds = %165, %152, %139, %126, %113, %100, %87, %74, %61, %48, %35
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #5

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
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
declare noundef i32 @vfprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ptr dead_on_return noundef) local_unnamed_addr #6

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!10 = !{!11}
!11 = distinct !{!11, !12, !"vprintf: argument 0"}
!12 = distinct !{!12, !"vprintf"}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 _ZTS8_IO_FILE", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
!16 = !{i64 0, i64 8, !17, i64 8, i64 8, !17, i64 16, i64 8, !17, i64 24, i64 4, !6, i64 28, i64 4, !6}
!17 = !{!15, !15, i64 0}
