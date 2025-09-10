; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/abs-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/abs-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca i64, align 8
  %18 = alloca i64, align 8
  %19 = alloca i64, align 8
  %20 = alloca i64, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store volatile i32 0, ptr %1, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile i32 1, ptr %2, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  store volatile i32 -1, ptr %3, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  store volatile i32 -2147483647, ptr %4, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  store volatile i32 2147483647, ptr %5, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store volatile i64 0, ptr %6, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  store volatile i64 1, ptr %7, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  store volatile i64 -1, ptr %8, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store volatile i64 -9223372036854775807, ptr %9, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  store volatile i64 9223372036854775807, ptr %10, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  store volatile i64 0, ptr %11, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store volatile i64 1, ptr %12, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  store volatile i64 -1, ptr %13, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  store volatile i64 -9223372036854775807, ptr %14, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store volatile i64 9223372036854775807, ptr %15, align 8, !tbaa !12
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  store volatile i64 0, ptr %16, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  store volatile i64 1, ptr %17, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store volatile i64 -1, ptr %18, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  store volatile i64 -9223372036854775807, ptr %19, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  store volatile i64 9223372036854775807, ptr %20, align 8, !tbaa !10
  %21 = load volatile i32, ptr %1, align 4, !tbaa !6
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %24, label %23

23:                                               ; preds = %0
  tail call void @abort() #5
  unreachable

24:                                               ; preds = %0
  %25 = load volatile i32, ptr %2, align 4, !tbaa !6
  %26 = tail call i32 @llvm.abs.i32(i32 %25, i1 true)
  %27 = icmp eq i32 %26, 1
  br i1 %27, label %29, label %28

28:                                               ; preds = %24
  tail call void @abort() #5
  unreachable

29:                                               ; preds = %24
  %30 = load volatile i32, ptr %3, align 4, !tbaa !6
  %31 = tail call i32 @llvm.abs.i32(i32 %30, i1 true)
  %32 = icmp eq i32 %31, 1
  br i1 %32, label %34, label %33

33:                                               ; preds = %29
  tail call void @abort() #5
  unreachable

34:                                               ; preds = %29
  %35 = load volatile i32, ptr %4, align 4, !tbaa !6
  %36 = tail call i32 @llvm.abs.i32(i32 %35, i1 true)
  %37 = icmp eq i32 %36, 2147483647
  br i1 %37, label %39, label %38

38:                                               ; preds = %34
  tail call void @abort() #5
  unreachable

39:                                               ; preds = %34
  %40 = load volatile i32, ptr %5, align 4, !tbaa !6
  %41 = tail call i32 @llvm.abs.i32(i32 %40, i1 true)
  %42 = icmp eq i32 %41, 2147483647
  br i1 %42, label %44, label %43

43:                                               ; preds = %39
  tail call void @abort() #5
  unreachable

44:                                               ; preds = %39
  %45 = load volatile i64, ptr %6, align 8, !tbaa !10
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %48, label %47

47:                                               ; preds = %44
  tail call void @abort() #5
  unreachable

48:                                               ; preds = %44
  %49 = load volatile i64, ptr %7, align 8, !tbaa !10
  %50 = tail call i64 @llvm.abs.i64(i64 %49, i1 true)
  %51 = icmp eq i64 %50, 1
  br i1 %51, label %53, label %52

52:                                               ; preds = %48
  tail call void @abort() #5
  unreachable

53:                                               ; preds = %48
  %54 = load volatile i64, ptr %8, align 8, !tbaa !10
  %55 = tail call i64 @llvm.abs.i64(i64 %54, i1 true)
  %56 = icmp eq i64 %55, 1
  br i1 %56, label %58, label %57

57:                                               ; preds = %53
  tail call void @abort() #5
  unreachable

58:                                               ; preds = %53
  %59 = load volatile i64, ptr %9, align 8, !tbaa !10
  %60 = tail call i64 @llvm.abs.i64(i64 %59, i1 true)
  %61 = icmp eq i64 %60, 9223372036854775807
  br i1 %61, label %63, label %62

62:                                               ; preds = %58
  tail call void @abort() #5
  unreachable

63:                                               ; preds = %58
  %64 = load volatile i64, ptr %10, align 8, !tbaa !10
  %65 = tail call i64 @llvm.abs.i64(i64 %64, i1 true)
  %66 = icmp eq i64 %65, 9223372036854775807
  br i1 %66, label %68, label %67

67:                                               ; preds = %63
  tail call void @abort() #5
  unreachable

68:                                               ; preds = %63
  %69 = load volatile i64, ptr %11, align 8, !tbaa !12
  %70 = icmp eq i64 %69, 0
  br i1 %70, label %72, label %71

71:                                               ; preds = %68
  tail call void @abort() #5
  unreachable

72:                                               ; preds = %68
  %73 = load volatile i64, ptr %12, align 8, !tbaa !12
  %74 = tail call i64 @llvm.abs.i64(i64 %73, i1 true)
  %75 = icmp eq i64 %74, 1
  br i1 %75, label %77, label %76

76:                                               ; preds = %72
  tail call void @abort() #5
  unreachable

77:                                               ; preds = %72
  %78 = load volatile i64, ptr %13, align 8, !tbaa !12
  %79 = tail call i64 @llvm.abs.i64(i64 %78, i1 true)
  %80 = icmp eq i64 %79, 1
  br i1 %80, label %82, label %81

81:                                               ; preds = %77
  tail call void @abort() #5
  unreachable

82:                                               ; preds = %77
  %83 = load volatile i64, ptr %14, align 8, !tbaa !12
  %84 = tail call i64 @llvm.abs.i64(i64 %83, i1 true)
  %85 = icmp eq i64 %84, 9223372036854775807
  br i1 %85, label %87, label %86

86:                                               ; preds = %82
  tail call void @abort() #5
  unreachable

87:                                               ; preds = %82
  %88 = load volatile i64, ptr %15, align 8, !tbaa !12
  %89 = tail call i64 @llvm.abs.i64(i64 %88, i1 true)
  %90 = icmp eq i64 %89, 9223372036854775807
  br i1 %90, label %92, label %91

91:                                               ; preds = %87
  tail call void @abort() #5
  unreachable

92:                                               ; preds = %87
  %93 = load volatile i64, ptr %16, align 8, !tbaa !10
  %94 = tail call i64 @imaxabs(i64 noundef %93) #6
  %95 = icmp eq i64 %94, 0
  br i1 %95, label %97, label %96

96:                                               ; preds = %92
  tail call void @abort() #5
  unreachable

97:                                               ; preds = %92
  %98 = tail call i64 @imaxabs(i64 noundef 0) #6
  %99 = icmp eq i64 %98, 0
  br i1 %99, label %101, label %100

100:                                              ; preds = %97
  tail call void @link_error() #6
  br label %101

101:                                              ; preds = %100, %97
  %102 = load volatile i64, ptr %17, align 8, !tbaa !10
  %103 = tail call i64 @imaxabs(i64 noundef %102) #6
  %104 = icmp eq i64 %103, 1
  br i1 %104, label %106, label %105

105:                                              ; preds = %101
  tail call void @abort() #5
  unreachable

106:                                              ; preds = %101
  %107 = tail call i64 @imaxabs(i64 noundef 1) #6
  %108 = icmp eq i64 %107, 1
  br i1 %108, label %110, label %109

109:                                              ; preds = %106
  tail call void @link_error() #6
  br label %110

110:                                              ; preds = %109, %106
  %111 = load volatile i64, ptr %18, align 8, !tbaa !10
  %112 = tail call i64 @imaxabs(i64 noundef %111) #6
  %113 = icmp eq i64 %112, 1
  br i1 %113, label %115, label %114

114:                                              ; preds = %110
  tail call void @abort() #5
  unreachable

115:                                              ; preds = %110
  %116 = tail call i64 @imaxabs(i64 noundef -1) #6
  %117 = icmp eq i64 %116, 1
  br i1 %117, label %119, label %118

118:                                              ; preds = %115
  tail call void @link_error() #6
  br label %119

119:                                              ; preds = %118, %115
  %120 = load volatile i64, ptr %19, align 8, !tbaa !10
  %121 = tail call i64 @imaxabs(i64 noundef %120) #6
  %122 = icmp eq i64 %121, 9223372036854775807
  br i1 %122, label %124, label %123

123:                                              ; preds = %119
  tail call void @abort() #5
  unreachable

124:                                              ; preds = %119
  %125 = tail call i64 @imaxabs(i64 noundef -9223372036854775807) #6
  %126 = icmp eq i64 %125, 9223372036854775807
  br i1 %126, label %128, label %127

127:                                              ; preds = %124
  tail call void @link_error() #6
  br label %128

128:                                              ; preds = %127, %124
  %129 = load volatile i64, ptr %20, align 8, !tbaa !10
  %130 = tail call i64 @imaxabs(i64 noundef %129) #6
  %131 = icmp eq i64 %130, 9223372036854775807
  br i1 %131, label %133, label %132

132:                                              ; preds = %128
  tail call void @abort() #5
  unreachable

133:                                              ; preds = %128
  %134 = tail call i64 @imaxabs(i64 noundef 9223372036854775807) #6
  %135 = icmp eq i64 %134, 9223372036854775807
  br i1 %135, label %137, label %136

136:                                              ; preds = %133
  tail call void @link_error() #6
  br label %137

137:                                              ; preds = %136, %133
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.abs.i32(i32, i1 immarg) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

declare void @link_error() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.abs.i64(i64, i1 immarg) #2

declare i64 @imaxabs(i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"long long", !8, i64 0}
