; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/initp1.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/initp1.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%class.Two = type { i32, i32, i32 }

@foo = dso_local local_unnamed_addr global %class.Two { i32 5, i32 6, i32 6 }, align 4
@goo = dso_local local_unnamed_addr global %class.Two { i32 7, i32 8, i32 7 }, align 4
@doo = dso_local local_unnamed_addr global [3 x %class.Two] [%class.Two { i32 0, i32 0, i32 14 }, %class.Two { i32 0, i32 0, i32 15 }, %class.Two { i32 0, i32 0, i32 16 }], align 4
@hoo = dso_local local_unnamed_addr global [3 x %class.Two] [%class.Two { i32 11, i32 12, i32 17 }, %class.Two { i32 13, i32 14, i32 18 }, %class.Two { i32 15, i32 16, i32 19 }], align 4
@coo = dso_local local_unnamed_addr global [3 x %class.Two] [%class.Two zeroinitializer, %class.Two { i32 0, i32 0, i32 1 }, %class.Two { i32 0, i32 0, i32 2 }], align 4
@koo = dso_local local_unnamed_addr global [3 x %class.Two] [%class.Two { i32 21, i32 22, i32 3 }, %class.Two { i32 23, i32 24, i32 4 }, %class.Two { i32 25, i32 26, i32 5 }], align 4
@xoo = dso_local local_unnamed_addr global [3 x %class.Two] [%class.Two { i32 0, i32 0, i32 8 }, %class.Two { i32 0, i32 0, i32 9 }, %class.Two { i32 0, i32 0, i32 10 }], align 4
@zoo = dso_local local_unnamed_addr global [3 x %class.Two] [%class.Two { i32 31, i32 32, i32 11 }, %class.Two { i32 33, i32 34, i32 12 }, %class.Two { i32 35, i32 36, i32 13 }], align 4
@_ZN3Two5countE = dso_local local_unnamed_addr global i32 20, align 4
@x = dso_local local_unnamed_addr global i64 0, align 8
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i64, ptr @x, align 8, !tbaa !6
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @coo, i64 8), align 4, !tbaa !10
  %3 = zext nneg i32 %2 to i64
  %4 = shl nuw i64 1, %3
  %5 = and i64 %4, %1
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %153

7:                                                ; preds = %0
  %8 = or i64 %4, %1
  store i64 %8, ptr @x, align 8, !tbaa !6
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @coo, i64 20), align 4, !tbaa !10
  %10 = zext nneg i32 %9 to i64
  %11 = shl nuw i64 1, %10
  %12 = and i64 %11, %8
  %13 = icmp eq i64 %12, 0
  br i1 %13, label %14, label %153

14:                                               ; preds = %7
  %15 = or i64 %11, %8
  store i64 %15, ptr @x, align 8, !tbaa !6
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @coo, i64 32), align 4, !tbaa !10
  %17 = zext nneg i32 %16 to i64
  %18 = shl nuw i64 1, %17
  %19 = and i64 %18, %15
  %20 = icmp eq i64 %19, 0
  br i1 %20, label %21, label %153

21:                                               ; preds = %14
  %22 = or i64 %18, %15
  store i64 %22, ptr @x, align 8, !tbaa !6
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @koo, i64 8), align 4, !tbaa !10
  %24 = zext nneg i32 %23 to i64
  %25 = shl nuw i64 1, %24
  %26 = and i64 %25, %22
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %28, label %153

28:                                               ; preds = %21
  %29 = or i64 %25, %22
  store i64 %29, ptr @x, align 8, !tbaa !6
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @koo, i64 20), align 4, !tbaa !10
  %31 = zext nneg i32 %30 to i64
  %32 = shl nuw i64 1, %31
  %33 = and i64 %32, %29
  %34 = icmp eq i64 %33, 0
  br i1 %34, label %35, label %153

35:                                               ; preds = %28
  %36 = or i64 %32, %29
  store i64 %36, ptr @x, align 8, !tbaa !6
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @koo, i64 32), align 4, !tbaa !10
  %38 = zext nneg i32 %37 to i64
  %39 = shl nuw i64 1, %38
  %40 = and i64 %39, %36
  %41 = icmp eq i64 %40, 0
  br i1 %41, label %42, label %153

42:                                               ; preds = %35
  %43 = or i64 %39, %36
  store i64 %43, ptr @x, align 8, !tbaa !6
  %44 = icmp eq i64 %43, 63
  br i1 %44, label %46, label %45

45:                                               ; preds = %42
  tail call void @abort() #3
  unreachable

46:                                               ; preds = %42
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @foo, i64 8), align 4, !tbaa !10
  %48 = icmp ugt i32 %47, 5
  br i1 %48, label %49, label %153

49:                                               ; preds = %46
  %50 = zext nneg i32 %47 to i64
  %51 = shl nuw i64 1, %50
  %52 = or i64 %51, 63
  store i64 %52, ptr @x, align 8, !tbaa !6
  %53 = icmp eq i64 %52, 127
  br i1 %53, label %55, label %54

54:                                               ; preds = %49
  tail call void @abort() #3
  unreachable

55:                                               ; preds = %49
  %56 = load i32, ptr getelementptr inbounds nuw (i8, ptr @goo, i64 8), align 4, !tbaa !10
  %57 = icmp ugt i32 %56, 6
  br i1 %57, label %58, label %153

58:                                               ; preds = %55
  %59 = zext nneg i32 %56 to i64
  %60 = shl nuw i64 1, %59
  %61 = or i64 %60, 127
  store i64 %61, ptr @x, align 8, !tbaa !6
  %62 = icmp eq i64 %61, 255
  br i1 %62, label %64, label %63

63:                                               ; preds = %58
  tail call void @abort() #3
  unreachable

64:                                               ; preds = %58
  %65 = load i32, ptr getelementptr inbounds nuw (i8, ptr @xoo, i64 8), align 4, !tbaa !10
  %66 = icmp ugt i32 %65, 7
  br i1 %66, label %67, label %153

67:                                               ; preds = %64
  %68 = zext nneg i32 %65 to i64
  %69 = shl nuw i64 1, %68
  %70 = or i64 %69, 255
  store i64 %70, ptr @x, align 8, !tbaa !6
  %71 = load i32, ptr getelementptr inbounds nuw (i8, ptr @xoo, i64 20), align 4, !tbaa !10
  %72 = zext nneg i32 %71 to i64
  %73 = shl nuw i64 1, %72
  %74 = and i64 %73, %70
  %75 = icmp eq i64 %74, 0
  br i1 %75, label %76, label %153

76:                                               ; preds = %67
  %77 = or i64 %73, %70
  store i64 %77, ptr @x, align 8, !tbaa !6
  %78 = load i32, ptr getelementptr inbounds nuw (i8, ptr @xoo, i64 32), align 4, !tbaa !10
  %79 = zext nneg i32 %78 to i64
  %80 = shl nuw i64 1, %79
  %81 = and i64 %80, %77
  %82 = icmp eq i64 %81, 0
  br i1 %82, label %83, label %153

83:                                               ; preds = %76
  %84 = or i64 %80, %77
  store i64 %84, ptr @x, align 8, !tbaa !6
  %85 = load i32, ptr getelementptr inbounds nuw (i8, ptr @zoo, i64 8), align 4, !tbaa !10
  %86 = zext nneg i32 %85 to i64
  %87 = shl nuw i64 1, %86
  %88 = and i64 %87, %84
  %89 = icmp eq i64 %88, 0
  br i1 %89, label %90, label %153

90:                                               ; preds = %83
  %91 = or i64 %87, %84
  store i64 %91, ptr @x, align 8, !tbaa !6
  %92 = load i32, ptr getelementptr inbounds nuw (i8, ptr @zoo, i64 20), align 4, !tbaa !10
  %93 = zext nneg i32 %92 to i64
  %94 = shl nuw i64 1, %93
  %95 = and i64 %94, %91
  %96 = icmp eq i64 %95, 0
  br i1 %96, label %97, label %153

97:                                               ; preds = %90
  %98 = or i64 %94, %91
  store i64 %98, ptr @x, align 8, !tbaa !6
  %99 = load i32, ptr getelementptr inbounds nuw (i8, ptr @zoo, i64 32), align 4, !tbaa !10
  %100 = zext nneg i32 %99 to i64
  %101 = shl nuw i64 1, %100
  %102 = and i64 %101, %98
  %103 = icmp eq i64 %102, 0
  br i1 %103, label %104, label %153

104:                                              ; preds = %97
  %105 = or i64 %101, %98
  store i64 %105, ptr @x, align 8, !tbaa !6
  %106 = icmp eq i64 %105, 16383
  br i1 %106, label %108, label %107

107:                                              ; preds = %104
  tail call void @abort() #3
  unreachable

108:                                              ; preds = %104
  %109 = load i32, ptr getelementptr inbounds nuw (i8, ptr @doo, i64 8), align 4, !tbaa !10
  %110 = icmp ugt i32 %109, 13
  br i1 %110, label %111, label %153

111:                                              ; preds = %108
  %112 = zext nneg i32 %109 to i64
  %113 = shl nuw i64 1, %112
  %114 = or i64 %113, 16383
  store i64 %114, ptr @x, align 8, !tbaa !6
  %115 = load i32, ptr getelementptr inbounds nuw (i8, ptr @doo, i64 20), align 4, !tbaa !10
  %116 = zext nneg i32 %115 to i64
  %117 = shl nuw i64 1, %116
  %118 = and i64 %117, %114
  %119 = icmp eq i64 %118, 0
  br i1 %119, label %120, label %153

120:                                              ; preds = %111
  %121 = or i64 %117, %114
  store i64 %121, ptr @x, align 8, !tbaa !6
  %122 = load i32, ptr getelementptr inbounds nuw (i8, ptr @doo, i64 32), align 4, !tbaa !10
  %123 = zext nneg i32 %122 to i64
  %124 = shl nuw i64 1, %123
  %125 = and i64 %124, %121
  %126 = icmp eq i64 %125, 0
  br i1 %126, label %127, label %153

127:                                              ; preds = %120
  %128 = or i64 %124, %121
  store i64 %128, ptr @x, align 8, !tbaa !6
  %129 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hoo, i64 8), align 4, !tbaa !10
  %130 = zext nneg i32 %129 to i64
  %131 = shl nuw i64 1, %130
  %132 = and i64 %131, %128
  %133 = icmp eq i64 %132, 0
  br i1 %133, label %134, label %153

134:                                              ; preds = %127
  %135 = or i64 %131, %128
  store i64 %135, ptr @x, align 8, !tbaa !6
  %136 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hoo, i64 20), align 4, !tbaa !10
  %137 = zext nneg i32 %136 to i64
  %138 = shl nuw i64 1, %137
  %139 = and i64 %138, %135
  %140 = icmp eq i64 %139, 0
  br i1 %140, label %141, label %153

141:                                              ; preds = %134
  %142 = or i64 %138, %135
  store i64 %142, ptr @x, align 8, !tbaa !6
  %143 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hoo, i64 32), align 4, !tbaa !10
  %144 = zext nneg i32 %143 to i64
  %145 = shl nuw i64 1, %144
  %146 = and i64 %145, %142
  %147 = icmp eq i64 %146, 0
  br i1 %147, label %148, label %153

148:                                              ; preds = %141
  %149 = or i64 %145, %142
  store i64 %149, ptr @x, align 8, !tbaa !6
  %150 = icmp eq i64 %149, 1048575
  br i1 %150, label %152, label %151

151:                                              ; preds = %148
  tail call void @abort() #3
  unreachable

152:                                              ; preds = %148
  tail call void @exit(i32 noundef 0) #4
  unreachable

153:                                              ; preds = %141, %134, %127, %120, %111, %108, %97, %90, %83, %76, %67, %64, %55, %46, %35, %28, %21, %14, %7, %0
  ret i32 1
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold noreturn nounwind }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !12, i64 8}
!11 = !{!"_ZTS3Two", !12, i64 0, !12, i64 4, !12, i64 8}
!12 = !{!"int", !8, i64 0}
